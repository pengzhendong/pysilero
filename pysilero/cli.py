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

from pysilero import init_session, SileroVAD, VADIterator


@click.command()
@click.argument("wav_path", type=click.Path(exists=True, file_okay=True))
@click.option("--streaming/--no-streaming", default=False, help="Streming mode")
@click.option("--save-path", help="Save path for output audio")
def main(wav_path: str, streaming: bool, save_path: str):
    session = init_session()
    sample_rate = sf.info(wav_path).samplerate
    model = SileroVAD(session, sample_rate, denoise=True)

    if not streaming:
        speech_timestamps = model.get_speech_timestamps(
            wav_path, return_seconds=True, save_path=save_path
        )
        print("None streaming result:", list(speech_timestamps))
    else:
        print("Streaming result:", end=" ")
        audio, sampling_rate = sf.read(wav_path, dtype=np.float32)
        vad_iterator = VADIterator(model)
        if save_path:
            out_wav = wave.open(save_path, "w")
            out_wav.setnchannels(1)
            out_wav.setsampwidth(2)
            out_wav.setframerate(sampling_rate)
        # number of samples in a single audio chunk, 10ms per chunk
        window_size_samples = 10 * sampling_rate // 1000
        for i in range(0, len(audio), window_size_samples):
            chunk = audio[i : i + window_size_samples]
            last = i + window_size_samples >= len(audio)
            for speech_dict, speech_samples in vad_iterator(
                chunk, last, return_seconds=True
            ):
                if "start" in speech_dict or "end" in speech_dict:
                    print(speech_dict, end=" ")
                if save_path and speech_samples is not None:
                    out_wav.writeframes((speech_samples * 32768).astype(np.int16))
        # reset after each audio
        vad_iterator.reset()


if __name__ == "__main__":
    main()
