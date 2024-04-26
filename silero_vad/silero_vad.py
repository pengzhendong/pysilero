# Copyright (c) 2024, Zhendong Peng (pzd17@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from functools import partial
from pathlib import Path
from typing import Union
import warnings

import librosa
import numpy as np
import soundfile as sf
import soxr

from .inference_session import PickableInferenceSession
from .utils import get_energy


class SileroVAD:
    def __init__(self, onnx_model=f"{os.path.dirname(__file__)}/silero_vad.onnx"):
        self.session = PickableInferenceSession(onnx_model)
        self.reset_states()
        self.sample_rates = [8000, 16000]

    def reset_states(self):
        self._h = np.zeros((2, 1, 64)).astype("float32")
        self._c = np.zeros((2, 1, 64)).astype("float32")

    def __call__(self, x, sr: int):
        ort_inputs = {
            "input": x[np.newaxis, :],
            "h": self._h,
            "c": self._c,
            "sr": np.array(sr, dtype="int64"),
        }
        ort_outs = self.session.run(None, ort_inputs)
        out, self._h, self._c = ort_outs
        return out

    def process_segment(
        self,
        idx,
        segment,
        wav,
        sample_rate,
        step,
        save_path,
        flat_layout,
        speech_pad_ms,
        return_seconds,
    ):
        if step != 1.0:
            segment["start"] = int(segment["start"] * step)
            segment["end"] = int(segment["end"] * step)

        speech_pad_samples = speech_pad_ms * sample_rate // 1000
        segment["start"] = max(segment["start"] - speech_pad_samples, 0)
        segment["end"] = min(segment["end"] + speech_pad_samples, len(wav))
        if save_path:
            wav = wav[segment["start"] : segment["end"]]
            if flat_layout:
                sf.write(str(save_path) + f"_{idx:04d}.wav", wav, sample_rate)
            else:
                sf.write(str(Path(save_path) / f"{idx:04d}.wav"), wav, sample_rate)
        if return_seconds:
            segment["start"] = round(segment["start"] / sample_rate, 3)
            segment["end"] = round(segment["end"] / sample_rate, 3)
        return segment

    def get_speech_timestamps(
        self,
        wav_path: Union[str, Path],
        save_path: Union[str, Path] = None,
        flat_layout: bool = True,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = float("inf"),
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        window_size_ms: int = 32,
        return_seconds: bool = False,
    ):
        """
        Splitting long audios into speech chunks using silero VAD

        Parameters
        ----------
        wav_path: wav path
        save_path: string or Path (default - None)
            whether the save speech segments
        flat_layout: bool (default - True)
            whether use the flat directory structure
        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each audio
            chunk, probabilities ABOVE this value are considered as SPEECH. It is
            better to tune this parameter for each dataset separately, but "lazy"
            0.5 is pretty good for most datasets.
        min_speech_duration_ms: int (default - 250 milliseconds)
            Final speech chunks shorter min_speech_duration_ms are thrown out
        max_speech_duration_s: int (default - inf)
            Maximum duration of speech chunks in seconds
            Chunks longer than max_speech_duration_s will be split at the timestamp
            of the last silence that lasts more than 98ms (if any), to prevent
            agressive cutting. Otherwise, they will be split aggressively just
            before max_speech_duration_s.
        min_silence_duration_ms: int (default - 300 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before
            separating it.
        speech_pad_ms: int (default - 100 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        window_size_ms: int (default - 32 milliseconds)
            Audio chunks of window_size_ms size are fed to the silero VAD model.
            WARNING! Silero VAD models were trained using 32, 64, 96 milliseconds
            for 8000 sample rate and 16000 sample rate.
            Values other than these may affect model perfomance!!
        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)

        Returns
        ----------
        speeches: list of dicts
            list containing ends and beginnings of speech chunks (samples or seconds
            based on return_seconds)
        """

        wav_path = Path(wav_path)
        original_sr = sf.info(wav_path).samplerate
        original_wav, _ = sf.read(wav_path, dtype="float32")
        if original_sr in self.sample_rates:
            step = 1.0
            wav, sr = original_wav, original_sr
        else:
            step = original_sr / 16000
            wav, sr = librosa.load(wav_path, sr=16000)

        fn = partial(
            self.process_segment,
            wav=original_wav,
            sample_rate=original_sr,
            step=step,
            save_path=save_path,
            flat_layout=flat_layout,
            speech_pad_ms=speech_pad_ms,
            return_seconds=return_seconds,
        )

        if len(wav.shape) > 1:
            raise ValueError("Only supported mono wav.")
        if len(wav) / sr * 1000 < 32:
            raise ValueError("Input audio is too short.")
        if window_size_ms not in [32, 64, 96]:
            warnings.warn(
                "Unusual window_size_ms! Supported window_size_ms: [32, 64, 96]"
            )
        speech_pad_samples = speech_pad_ms * sr // 1000
        window_size_samples = window_size_ms * sr // 1000
        min_silence_samples_at_max_speech = 98 * sr // 1000
        min_speech_samples = min_speech_duration_ms * sr // 1000
        min_silence_samples = min_silence_duration_ms * sr // 1000
        max_speech_duration_samples = max_speech_duration_s * sr
        max_speech_samples = (
            max_speech_duration_samples - window_size_samples - 2 * speech_pad_samples
        )

        self.reset_states()
        num_samples = len(wav)

        idx = 0
        current_speech = {}
        neg_threshold = threshold - 0.15
        triggered = False
        # to save potential segment end (and tolerate some silence)
        temp_end = 0
        # to save potential segment limits in case of maximum segment size reached
        prev_end = 0
        next_start = 0
        for current_samples in range(0, num_samples, window_size_samples):
            chunk = wav[current_samples : current_samples + window_size_samples]
            if len(chunk) < window_size_samples:
                chunk = np.pad(chunk, (0, int(window_size_samples - len(chunk))))
            speech_prob = self(chunk, sr)

            # current frame is speech
            if speech_prob >= threshold:
                if temp_end > 0 and next_start < prev_end:
                    next_start = current_samples
                temp_end = 0
                if not triggered:
                    triggered = True
                    current_speech["start"] = current_samples
                    continue
            # in speech, and speech duration is more than max speech duration
            if (
                triggered
                and current_samples - current_speech["start"] > max_speech_samples
            ):
                # prev_end larger than 0 means there is a short silence in the middle avoid aggressive cutting
                if prev_end > 0:
                    current_speech["end"] = prev_end
                    yield fn(idx, current_speech)
                    idx += 1
                    current_speech = {}
                    # previously reached silence (< neg_thres) and is still not speech (< thres)
                    if next_start < prev_end:
                        triggered = False
                    else:
                        current_speech["start"] = next_start
                    prev_end = 0
                    next_start = 0
                    temp_end = 0
                else:
                    current_speech["end"] = current_samples
                    yield fn(idx, current_speech)
                    idx += 1
                    current_speech = {}
                    prev_end = 0
                    next_start = 0
                    temp_end = 0
                    triggered = False
                    continue
            # in speech, and current frame is silence
            if triggered and speech_prob < neg_threshold:
                if temp_end == 0:
                    temp_end = current_samples
                # record the last silence before reaching max speech duration
                if current_samples - temp_end > min_silence_samples_at_max_speech:
                    prev_end = temp_end
                if current_samples - temp_end >= min_silence_samples:
                    current_speech["end"] = temp_end
                    # keep the speech segment if it is longer than min_speech_samples
                    if (
                        current_speech["end"] - current_speech["start"]
                        > min_speech_samples
                    ):
                        yield fn(idx, current_speech)
                        idx += 1
                    current_speech = {}
                    prev_end = 0
                    next_start = 0
                    temp_end = 0
                    triggered = False

        # deal with the last speech segment
        if (
            current_speech
            and num_samples - current_speech["start"] > min_speech_samples
        ):
            current_speech["end"] = num_samples
            yield fn(idx, current_speech)
            idx += 1


class VADIterator:
    def __init__(
        self,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 100,
        window_size_ms: int = 32,
    ):
        """
        Class for stream imitation

        Parameters
        ----------
        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each
            audio chunk, probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately, but
            "lazy" 0.5 is pretty good for most datasets.
        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 sample rates
        min_silence_duration_ms: int (default - 300 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms
            before separating it
        speech_pad_ms: int (default - 100 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        window_size_ms: int (default - 32 milliseconds)
            Audio chunks of window_size_ms size are fed to the silero VAD model.
            WARNING! Silero VAD models were trained using 32, 64, 96 milliseconds
            for 8000 sample rate and 16000 sample rate.
            Values other than these may affect model perfomance!!
        """

        self.model = SileroVAD()
        if window_size_ms not in [32, 64, 96]:
            warnings.warn(
                "Unusual window_size_ms! Supported window_size_ms: [32, 64, 96]"
            )
        if sampling_rate not in self.model.sample_rates:
            self.resampler = soxr.ResampleStream(
                sampling_rate, 16000, 1, dtype=np.int16
            )
            sampling_rate = 16000
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.speech_pad_samples = speech_pad_ms * sampling_rate // 1000
        self.window_size_samples = window_size_ms * sampling_rate // 1000
        self.min_silence_samples = sampling_rate * min_silence_duration_ms // 1000
        self.reset_states()

    def reset_states(self):
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0
        self.remained_samples = np.empty(0, dtype=np.float32)

    def __call__(self, x, return_seconds=False):
        """
        x: audio chunk

        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)
        """

        self.remained_samples = np.concatenate((self.remained_samples, x), axis=0)
        if self.remained_samples.shape[0] < self.window_size_samples:
            return
        x = self.remained_samples[: self.window_size_samples]
        self.remained_samples = self.remained_samples[self.window_size_samples :]
        self.current_sample += self.window_size_samples
        speech_prob = self.model(x, self.sampling_rate)
        # Suppress background vocals by harmonic energy
        # energy = get_energy(x, self.sampling_rate, from_harmonic=4)
        # if speech_prob < 0.9 and energy < 500 * (1 - speech_prob):
        #     speech_prob = 0

        if speech_prob >= self.threshold:
            self.temp_end = 0
            # triggered = True means the speech has been started
            if not self.triggered:
                self.triggered = True
                speech_start = (
                    self.current_sample
                    - self.window_size_samples
                    - self.speech_pad_samples
                )
                if return_seconds:
                    speech_start = round(speech_start / self.sampling_rate, 3)
                return {"start": speech_start}

        if speech_prob < self.threshold - 0.15 and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end >= self.min_silence_samples:
                speech_end = self.temp_end + self.speech_pad_samples
                if return_seconds:
                    speech_end = round(speech_end / self.sampling_rate, 3)
                self.temp_end = 0
                self.triggered = False
                return {"end": speech_end}
        return
