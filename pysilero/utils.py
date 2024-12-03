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

import warnings

import librosa
import numpy as np
import parselmouth

warnings.filterwarnings("ignore")


def get_energy(chunk, sr, from_harmonic=1, to_harmonic=5):
    sound = parselmouth.Sound(chunk, sampling_frequency=sr)
    # pitch
    pitch = sound.to_pitch(pitch_floor=100, pitch_ceiling=350)
    # pitch energy
    # energy = np.mean(pitch.selected_array["strength"])
    pitch = np.mean(pitch.selected_array["frequency"])
    # frame log energy
    # energy = np.mean(sound.to_mfcc().to_array(), axis=1)[0]

    # energy form x-th harmonic to y-th harmonic
    freqs = librosa.fft_frequencies(sr=sr)
    freq_band_idx = np.where((freqs >= from_harmonic * pitch) & (freqs <= to_harmonic * pitch))[0]
    energy = np.sum(np.abs(librosa.stft(chunk)[freq_band_idx, :]))

    return energy
