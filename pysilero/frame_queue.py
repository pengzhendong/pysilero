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
    def __init__(self, frame_size, in_rate, speech_pad_samples=0, out_rate=None, padding=True):
        self.frame_size = frame_size
        # padding zeros for the last frame
        self.padding = padding
        self.speech_pad_samples = speech_pad_samples
        # cache the original samples for padding and soxr's delay
        # TODO: use the largest delay of soxr instead of 500ms cache
        num_cached_samples = speech_pad_samples + 500 * in_rate // 1000
        self.cached_samples = np.zeros(num_cached_samples, dtype=np.float32)
        self.cache_start = -len(self.cached_samples)

        self.current_sample = 0
        self.remained_samples = np.empty(0, dtype=np.float32)

        if out_rate is None or in_rate == out_rate:
            self.step = 1.0
            self.resampler = None
        else:
            self.step = in_rate / out_rate
            self.resampler = soxr.ResampleStream(in_rate, out_rate, num_channels=1)

    def add_chunk(self, chunk, is_last=False):
        # cache the original frame without resampling for `lookforward` of vad start
        # cache start is the absolute sample index of the first sample in the cached_samples
        if len(chunk) > 0:
            self.cache_start += len(chunk)
            self.cached_samples = np.roll(self.cached_samples, -len(chunk))
            self.cached_samples[-len(chunk) :] = chunk[-len(self.cached_samples) :]
            # resample
            if self.resampler is not None:
                chunk = self.resampler.resample_chunk(chunk, is_last)
            # enqueue chunk
            self.remained_samples = np.concatenate((self.remained_samples, chunk))

        while len(self.remained_samples) >= self.frame_size:
            frame = self.remained_samples[: self.frame_size]
            self.remained_samples = self.remained_samples[self.frame_size :]
            # frame_start and frame_end is the sample index before resampling
            frame_start = self.current_sample
            self.current_sample += int(len(frame) * self.step)
            frame_end = self.current_sample
            yield frame_start, frame_end, frame

        if is_last and len(self.remained_samples) > 0 and self.padding:
            frame = self.remained_samples
            frame_start = self.current_sample
            self.current_sample += int(len(frame) * self.step)
            frame = np.pad(frame, (0, self.frame_size - len(frame)))
            frame_end = self.current_sample
            yield frame_start, frame_end, frame

    def get_frame(self, speech_padding=False):
        # dequeue one original frame without resampling
        frame_start = self.current_sample - int(self.frame_size * self.step)
        frame_end = self.current_sample
        if speech_padding:
            frame_start -= self.speech_pad_samples
        # get the relative sample index of the speech
        speech_start = frame_start - self.cache_start
        speech_end = frame_end - self.cache_start
        return self.cached_samples[speech_start:speech_end]


if __name__ == "__main__":
    queue = FrameQueue(3, 1000)
    frames = [[1, 2, 3], [4, 5], [6, 7, 8]]
    for index, frame in enumerate(frames):
        for frame_start, frame_end, frame in queue.add_chunk(frame, index == len(frames) - 1):
            print(frame_start, frame_end, frame)
