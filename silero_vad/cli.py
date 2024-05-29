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
import numpy as np
import soundfile as sf
import wave

from silero_vad import SileroVAD, VADIterator


@click.command()
@click.argument("wav_path", type=click.Path(exists=True, file_okay=True))
@click.option("--streaming/--no-streaming", default=False, help="Streming mode")
@click.option("--save-path", help="Save path for output audio")
def main(wav_path: str, streaming: bool, save_path: str):
    if not streaming:
        vad = SileroVAD()
        speech_timestamps = vad.get_speech_timestamps(
            wav_path, return_seconds=True, save_path=save_path
        )
        print("None streaming result:", list(speech_timestamps))
    else:
        print("Streaming result:", end=" ")
        wav, sr = sf.read(wav_path, dtype=np.float32)
        vad_iterator = VADIterator(sampling_rate=sr)

        if save_path:
            out_wav = wave.open(save_path, "w")
            out_wav.setnchannels(1)
            out_wav.setsampwidth(2)
            out_wav.setframerate(sr)
        # number of samples in a single audio chunk, 10ms per chunk
        window_size_samples = 10 * sr // 1000
        for i in range(0, len(wav), window_size_samples):
            chunk = wav[i : i + window_size_samples]
            last = len(chunk) < window_size_samples
            for speech_dict, speech_samples in vad_iterator(
                chunk, last, return_seconds=True
            ):
                if speech_dict:
                    print(speech_dict, end=" ")
                if save_path and speech_samples is not None:
                    out_wav.writeframes((speech_samples * 32768).astype(np.int16))
        # reset model states after each audio
        vad_iterator.reset_states()


if __name__ == "__main__":
    main()
