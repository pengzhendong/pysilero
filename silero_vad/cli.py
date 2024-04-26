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
import soxr

from silero_vad import SileroVAD, VADIterator


@click.command()
@click.argument("wav_path", type=click.Path(exists=True, file_okay=True))
@click.option("--streaming/--no-streaming", default=False, help="Streming mode")
def main(wav_path: str, streaming: bool):
    if not streaming:
        vad = SileroVAD()
        speech_timestamps = vad.get_speech_timestamps(
            wav_path,
            min_silence_duration_ms=300,
            speech_pad_ms=100,
            return_seconds=True,
        )
        print("None streaming result:", list(speech_timestamps))
    else:
        print("Streaming result:", end=" ")
        wav, sr = sf.read(wav_path, dtype=np.float32)
        vad_iterator = VADIterator(
            min_silence_duration_ms=300, speech_pad_ms=100, sampling_rate=sr
        )
        # number of samples in a single audio chunk
        window_size_samples = 32 * sr // 1000

        for i in range(0, len(wav), window_size_samples):
            chunk = wav[i : i + window_size_samples]
            if len(chunk) < window_size_samples:
                break
            speech_dict = vad_iterator(chunk, return_seconds=True)
            if speech_dict:
                print(speech_dict, end=" ")
        # reset model states after each audio
        vad_iterator.reset_states()


if __name__ == "__main__":
    main()
