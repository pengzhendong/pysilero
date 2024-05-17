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

import numpy as np
import soxr


class FrameQueue:
    def __init__(
        self, frame_size_ms, src_sr, speech_pad_ms=0, dst_sr=None, padding=True
    ):
        # padding zeros for the last frame
        self.padding = padding
        # cache the original samples for padding and soxr's delay
        self.speech_pad_samples = speech_pad_ms * src_sr // 1000
        # TODO: use the largest delay of soxr
        self.cached_ms = speech_pad_ms + 500
        self.current_sample = 0
        self.remained_samples = np.empty(0, dtype=np.float32)

        if self.cached_ms > 0:
            self.cached_samples = np.zeros(self.cached_ms * src_sr // 1000)
            self.cache_start = -len(self.cached_samples)

        if src_sr == dst_sr or dst_sr is None:
            self.step = 1.0
            self.resampler = None
            self.frame_size = frame_size_ms * src_sr // 1000
        else:
            self.step = src_sr / dst_sr
            self.frame_size = frame_size_ms * dst_sr // 1000
            self.resampler = soxr.ResampleStream(src_sr, dst_sr, 1)

    def add_chunk(self, chunk, last=False):
        # cache
        if self.cached_ms > 0:
            self.cache_start += len(chunk)
            self.cached_samples = np.roll(self.cached_samples, -len(chunk))
            self.cached_samples[-len(chunk) :] = chunk[-len(self.cached_samples) :]
        # resample
        if self.resampler is not None:
            chunk = self.resampler.resample_chunk(chunk, last)
        # enqueue chunk
        self.remained_samples = np.concatenate((self.remained_samples, chunk))
        while len(self.remained_samples) >= self.frame_size:
            frame = self.remained_samples[: self.frame_size]
            self.remained_samples = self.remained_samples[self.frame_size :]
            frame_start = self.current_sample
            self.current_sample += len(frame)
            yield frame_start, self.current_sample, frame

        if last and len(self.remained_samples) > 0 and self.padding:
            frame = self.remained_samples
            frame_start = self.current_sample
            self.current_sample += len(frame)
            frame = np.pad(frame, (0, self.frame_size - len(frame)))
            yield frame_start, self.current_sample, frame

    def get_original_frame(self, speech_padding=False):
        frame_start = self.current_sample - self.frame_size
        if speech_padding:
            frame_start -= self.speech_pad_samples
        speech_start = int(frame_start * self.step) - self.cache_start
        speech_end = int(self.current_sample * self.step) - self.cache_start
        return self.cached_samples[speech_start:speech_end]

    def clear(self):
        if self.cached_ms > 0:
            self.cached_samples.fill(0)
            self.cache_start = -len(self.cached_samples)
        self.current_sample = 0
        self.remained_samples = np.empty(0, dtype=np.float32)


if __name__ == "__main__":
    queue = FrameQueue(3, 1000)
    frames = [[1, 2, 3], [4, 5], [6, 7, 8]]
    for idx, frame in enumerate(frames):
        for frame_start, frame_end, frame in queue.add_chunk(
            frame, idx == len(frames) - 1
        ):
            print(frame_start, frame_end, frame)
