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

import numpy as np
import soundfile as sf

from .frame_queue import FrameQueue
from .inference_session import PickableInferenceSession
from .utils import get_energy


class SileroVAD:
    def __init__(self, onnx_model=f"{os.path.dirname(__file__)}/silero_vad.onnx"):
        self.session = PickableInferenceSession(onnx_model)
        self.reset_states()
        self.sample_rates = [8000, 16000]

    def reset_states(self):
        self._h = np.zeros((2, 1, 64)).astype(np.float32)
        self._c = np.zeros((2, 1, 64)).astype(np.float32)

    def __call__(self, x, sr: int):
        ort_inputs = {
            "input": x[np.newaxis, :],
            "h": self._h,
            "c": self._c,
            "sr": np.array(sr, dtype=np.int64),
        }
        ort_outs = self.session.run(None, ort_inputs)
        out, self._h, self._c = ort_outs
        return out

    @staticmethod
    def process_segment(
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
                sf.write(str(save_path) + f"_{idx:05d}.wav", wav, sample_rate)
            else:
                save_path = Path(save_path)
                if not save_path.exists():
                    save_path.mkdir(parents=True, exist_ok=True)
                sf.write(str(save_path / f"{idx:05d}.wav"), wav, sample_rate)
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

        wav, sr = sf.read(Path(wav_path), dtype=np.float32)
        if len(wav.shape) > 1:
            raise ValueError("Only supported mono wav.")
        if len(wav) / sr * 1000 < 32:
            raise ValueError("Input audio is too short.")
        if window_size_ms not in [32, 64, 96]:
            warnings.warn("Supported window_size_ms: [32, 64, 96]")

        if sr in self.sample_rates:
            vad_sr = sr
            queue = FrameQueue(window_size_ms, sr)
        else:
            vad_sr = 16000
            queue = FrameQueue(window_size_ms, src_sr=sr, dst_sr=vad_sr)

        fn = partial(
            self.process_segment,
            wav=wav,
            sample_rate=sr,
            step=queue.step,
            save_path=save_path,
            flat_layout=flat_layout,
            speech_pad_ms=speech_pad_ms,
            return_seconds=return_seconds,
        )

        speech_pad_samples = speech_pad_ms * vad_sr // 1000
        min_silence_samples_at_max_speech = 98 * vad_sr // 1000
        min_speech_samples = min_speech_duration_ms * vad_sr // 1000
        min_silence_samples = min_silence_duration_ms * vad_sr // 1000
        max_speech_duration_samples = max_speech_duration_s * vad_sr
        max_speech_samples = max_speech_duration_samples - 2 * speech_pad_samples
        self.reset_states()

        idx = 0
        current_speech = {}
        neg_threshold = threshold - 0.15
        triggered = False
        # to save potential segment end (and tolerate some silence)
        temp_end = 0
        # to save potential segment limits in case of maximum segment size reached
        prev_end = 0
        next_start = 0
        for frame_start, frame_end, frame in queue.add_chunk(wav):
            speech_prob = self(frame, vad_sr)
            # current frame is speech
            if speech_prob >= threshold:
                if temp_end > 0 and next_start < prev_end:
                    next_start = frame_end
                temp_end = 0
                if not triggered:
                    triggered = True
                    current_speech["start"] = frame_end
                    continue
            # in speech, and speech duration is more than max speech duration
            if triggered and frame_start - current_speech["start"] > max_speech_samples:
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
                    current_speech["end"] = frame_end
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
                    temp_end = frame_end
                # record the last silence before reaching max speech duration
                if frame_end - temp_end > min_silence_samples_at_max_speech:
                    prev_end = temp_end
                if frame_end - temp_end >= min_silence_samples:
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
        if current_speech and len(wav) - current_speech["start"] > min_speech_samples:
            current_speech["end"] = len(wav)
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
            warnings.warn("Supported window_size_ms: [32, 64, 96]")

        if sampling_rate in self.model.sample_rates:
            self.vad_sr = sampling_rate
            self.queue = FrameQueue(
                window_size_ms, sampling_rate, speech_pad_ms=speech_pad_ms
            )
        else:
            self.vad_sr = 16000
            self.queue = FrameQueue(
                window_size_ms,
                src_sr=sampling_rate,
                dst_sr=self.vad_sr,
                speech_pad_ms=speech_pad_ms,
            )

        self.threshold = threshold
        self.speech_pad_samples = speech_pad_ms * self.vad_sr // 1000
        self.min_silence_samples = min_silence_duration_ms * self.vad_sr // 1000
        self.reset_states()

    def reset_states(self):
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.queue.clear()

    def __call__(self, chunk, use_energy=False, return_seconds=False):
        """
        chunk: audio chunk

        use_energy: bool (default - False)
            whether to use harmonic energy to suppress background vocals
        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)
        """

        for frame_start, frame_end, frame in self.queue.add_chunk(chunk):
            speech_prob = self.model(frame, self.vad_sr)
            # Suppress background vocals by harmonic energy
            if use_energy:
                energy = get_energy(frame, self.vad_sr, from_harmonic=4)
                if speech_prob < 0.9 and energy < 500 * (1 - speech_prob):
                    speech_prob = 0

            if speech_prob >= self.threshold:
                self.temp_end = 0
                # triggered = True means the speech has been started
                if not self.triggered:
                    self.triggered = True
                    speech_start = frame_start - self.speech_pad_samples
                    if return_seconds:
                        speech_start = round(speech_start / self.vad_sr, 3)
                    yield {"start": speech_start}, self.queue.get_original_frame(True)
                else:
                    yield None, self.queue.get_original_frame()
            elif speech_prob < self.threshold - 0.15 and self.triggered:
                if not self.temp_end:
                    self.temp_end = frame_end
                if frame_end - self.temp_end >= self.min_silence_samples:
                    speech_end = self.temp_end + self.speech_pad_samples
                    if return_seconds:
                        speech_end = round(speech_end / self.vad_sr, 3)
                    self.temp_end = 0
                    self.triggered = False
                    yield {"end": speech_end}, self.queue.get_original_frame()
                else:
                    yield None, self.queue.get_original_frame()
