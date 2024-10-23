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

import click
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import wave

from pysilero import SileroVAD, VADIterator


@click.command()
@click.argument("wav_path", type=click.Path(exists=True, file_okay=True))
@click.option("--version", default="v5", help="Silero VAD version")
@click.option("--denoise/--no-denoise", default=False, help="Denoise before vad")
@click.option("--streaming/--no-streaming", default=False, help="Streming mode")
@click.option("--save-path", help="Save path for output audio")
@click.option("--plot/--no-plot", default=False, help="Plot the vad probabilities")
def main(wav_path, version, denoise, streaming, save_path, plot):
    sample_rate = sf.info(wav_path).samplerate

    if not streaming:
        model = SileroVAD(version, sample_rate, denoise=denoise)
        speech_timestamps = model.get_speech_timestamps(
            wav_path, return_seconds=True, save_path=save_path
        )
        print("None streaming result:", list(speech_timestamps))

        if plot:
            audio, sampling_rate = sf.read(wav_path, dtype=np.float32)
            x1 = np.arange(0, len(audio)) / sampling_rate
            outputs = list(model.get_speech_probs(wav_path))
            x2 = [i * 32 / 1000 for i in range(0, len(outputs))]
            plt.plot(x1, audio)
            plt.plot(x2, outputs)
            plt.show()
    else:
        print("Streaming result:", end=" ")
        audio, sampling_rate = sf.read(wav_path, dtype=np.float32)
        vad_iterator = VADIterator(version, sampling_rate)
        if save_path is not None:
            out_wav = wave.open(save_path, "w")
            out_wav.setnchannels(1)
            out_wav.setsampwidth(2)
            out_wav.setframerate(sampling_rate)
        # number of samples in a single audio chunk, 10ms per chunk
        window_size_samples = 10 * sampling_rate // 1000
        for i in range(0, len(audio), window_size_samples):
            chunk = audio[i : i + window_size_samples]
            is_last = i + window_size_samples >= len(audio)
            for speech_dict, speech_samples in vad_iterator(
                chunk, is_last, return_seconds=True
            ):
                if "start" in speech_dict or "end" in speech_dict:
                    print(speech_dict, end=" ")
                if save_path is not None and speech_samples is not None:
                    out_wav.writeframes((speech_samples * 32768).astype(np.int16))


if __name__ == "__main__":
    main()
