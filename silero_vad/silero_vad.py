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
from pathlib import Path
from typing import Union

import librosa
import numpy as np
import soundfile as sf
from loguru import logger

from .inference_session import PickableInferenceSession


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
            "input": x,
            "h": self._h,
            "c": self._c,
            "sr": np.array(sr, dtype="int64"),
        }
        ort_outs = self.session.run(None, ort_inputs)
        out, self._h, self._c = ort_outs
        return out

    def get_speech_timestamps(
        self,
        wav_path: Union[str, Path],
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = float("inf"),
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        window_size_samples: int = 512,
        return_seconds: bool = False,
        save_path: Union[str, Path] = None,
        flat_layout: bool = True,
    ):
        """
        Splitting long audios into speech chunks using silero VAD

        Parameters
        ----------
        wav_path: wav path
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
        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before
            separating it.
        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        window_size_samples: int (default - 1536 samples)
            Audio chunks of window_size_samples size are fed to the silero VAD model.
            WARNING! Silero VAD models were trained using 512, 1024, 1536 samples
            for 16000 sample rate and 256, 512, 768 samples for 8000 sample rate.
            Values other than these may affect model perfomance!!
        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)
        save_path: string or Path (default - None)
            whether the save speech segments
        flat_layout: bool (default - True)
            whether use the flat directory structure

        Returns
        ----------
        speeches: list of dicts
            list containing ends and beginnings of speech chunks (samples or seconds
            based on return_seconds)
        """

        wav_path = Path(wav_path)
        original_sr = librosa.get_samplerate(wav_path)
        if original_sr in self.sample_rates:
            step = 1
            wav, sr = librosa.load(wav_path, sr=original_sr)
        else:
            step = original_sr / 16000
            wav, sr = librosa.load(wav_path, sr=16000)
        if len(wav.shape) > 1:
            raise ValueError(
                "More than one dimension in audio."
                "Are you trying to process audio with 2 channels?"
            )
        if sr / len(wav) > 31.25:
            raise ValueError("Input audio is too short.")
        if window_size_samples not in [256, 512, 768, 1024, 1536]:
            logger.warning(
                "Unusual window_size_samples! Supported window_size_samples:"
                "\n - [512, 1024, 1536] for 16k sampling_rate"
                "\n - [256, 512, 768] for 8k sampling_rate"
            )
        if sr == 8000 and window_size_samples > 768:
            logger.warning(
                "window_size_samples is too big for 8k sampling rate!"
                "Better set window_size_samples to 256, 512 or 768 for 8k sample rate!"
            )
        min_speech_samples = sr * min_speech_duration_ms / 1000
        speech_pad_samples = sr * speech_pad_ms / 1000
        max_speech_samples = (
            sr * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples
        )
        min_silence_samples = sr * min_silence_duration_ms / 1000
        min_silence_samples_at_max_speech = sr * 98 / 1000

        self.reset_states()
        num_samples = len(wav)
        speech_probs = []
        for current_start_sample in range(0, num_samples, window_size_samples):
            chunk = wav[
                current_start_sample : current_start_sample + window_size_samples
            ]
            if len(chunk) < window_size_samples:
                chunk = np.pad(chunk, (0, int(window_size_samples - len(chunk))))
            speech_prob = self(chunk[np.newaxis, :], sr)
            speech_probs.append(speech_prob)

        speeches = []
        current_speech = {}
        neg_threshold = threshold - 0.15
        triggered = False
        # to save potential segment end (and tolerate some silence)
        temp_end = 0
        # to save potential segment limits in case of maximum segment size reached
        prev_end = 0
        next_start = 0
        for i, speech_prob in enumerate(speech_probs):
            current_samples = window_size_samples * i
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
                    speeches.append(current_speech)
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
                    speeches.append(current_speech)
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
                        speeches.append(current_speech)
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
            speeches.append(current_speech)

        # padding each speech segment
        for i, speech in enumerate(speeches):
            if i == 0:
                speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
            if i != len(speeches) - 1:
                silence_duration = speeches[i + 1]["start"] - speech["end"]
                if silence_duration < 2 * speech_pad_samples:
                    speech["end"] += int(silence_duration // 2)
                    speeches[i + 1]["start"] = int(
                        max(0, speeches[i + 1]["start"] - silence_duration // 2)
                    )
                else:
                    speech["end"] = int(
                        min(num_samples, speech["end"] + speech_pad_samples)
                    )
                    speeches[i + 1]["start"] = int(
                        max(0, speeches[i + 1]["start"] - speech_pad_samples)
                    )
            else:
                speech["end"] = int(
                    min(num_samples, speech["end"] + speech_pad_samples)
                )

        if step != 1.0:
            # reload without resampling
            wav, sr = librosa.load(wav_path, sr=original_sr)
        for idx, speech_dict in enumerate(speeches):
            if step != 1.0:
                speech_dict["start"] = int(speech_dict["start"] * step)
                speech_dict["end"] = int(speech_dict["end"] * step)
            if save_path:
                save_path = Path(save_path)
                segment = wav[speech_dict["start"] : speech_dict["end"]]
                if flat_layout:
                    sf.write(str(save_path) + f"_{idx:04d}.wav", segment, sr)
                else:
                    sf.write(str(save_path / f"{idx:04d}.wav"), segment, sr)
            if return_seconds:
                speech_dict["start"] = round(speech_dict["start"] / sr, 3)
                speech_dict["end"] = round(speech_dict["end"] / sr, 3)

        return speeches


class VADIterator:
    def __init__(
        self,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
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
        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms
            before separating it
        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        """

        self.model = SileroVAD()
        if sampling_rate not in self.model.sample_rates:
            raise ValueError(
                "VADIterator does not support sampling rates other than [8k, 16k]"
            )

        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(self, x, return_relative=False, return_seconds=False):
        """
        x: audio chunk

        return_relative: bool (default - False)
            whether return timestamps relative to the current audio chunk
        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)
        """

        window_size_samples = len(x)
        self.current_sample += window_size_samples
        speech_prob = self.model(x[np.newaxis, :], self.sampling_rate)

        if speech_prob >= self.threshold:
            self.temp_end = 0
            # triggered = True means the speech has been started
            if not self.triggered:
                self.triggered = True
                speech_start = (
                    self.current_sample - window_size_samples - self.speech_pad_samples
                )
                if return_relative:
                    speech_start = self.current_sample - speech_start
                if return_seconds:
                    speech_start = round(speech_start / self.sampling_rate, 3)
                return {"start": speech_start}
        if speech_prob < self.threshold - 0.15 and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end >= self.min_silence_samples:
                speech_end = self.temp_end + self.speech_pad_samples
                if return_relative:
                    speech_end = self.current_sample - speech_end
                if return_seconds:
                    speech_end = round(speech_end / self.sampling_rate, 3)
                self.temp_end = 0
                self.triggered = False
                return {"end": speech_end}
        return None