## Silero VAD

See [Silero VAD](https://github.com/snakers4/silero-vad).

### Denoiser

See [RnNoise](https://github.com/werman/noise-suppression-for-voice).

### Python Usage

``` bash
$ pip install https://github.com/pengzhendong/silero-vad/archive/refs/heads/master.zip
$ silero_vad --wav audio.wav
$ python
>>> from silero_vad import vad
>>> model = vad.OnnxWrapper()
>>> vad.get_speech_timestamps(model, "audio.wav")
```
