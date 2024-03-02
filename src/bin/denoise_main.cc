// Copyright (c) 2024 Zhendong Peng (pzd17@tsinghua.org.cn)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include "gflags/gflags.h"

#include "frontend/denoiser.h"
#include "frontend/resampler.h"
#include "frontend/wav.h"

DEFINE_string(input_wav_path, "", "input wav path");
DEFINE_string(output_wav_path, "", "output wav path");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  wav::WavReader wav_reader(FLAGS_input_wav_path);
  int num_channels = wav_reader.num_channels();
  CHECK_EQ(num_channels, 1) << "Only support mono (1 channel) wav!";
  int sample_rate = wav_reader.sample_rate();
  const float* pcm = wav_reader.data();
  int num_samples = wav_reader.num_samples();
  std::vector<float> input_pcm{pcm, pcm + num_samples};

  std::vector<float> output_pcm;
  // 0. Upsample to 48k for RnNoise
  if (sample_rate != 48000) {
    Resampler upsampler(sample_rate, 48000);
    upsampler.Resample(input_pcm, &output_pcm, true);
    input_pcm = output_pcm;
  }
  // 1. Denoise with RnNoise
  Denoiser denoiser;
  denoiser.Denoise(input_pcm, &output_pcm);
  // 2. Downsample back to original sample rate
  if (sample_rate != 48000) {
    input_pcm = output_pcm;
    Resampler downsampler(48000, sample_rate);
    downsampler.Resample(input_pcm, &output_pcm, true);
  }

  wav::WavWriter writer(output_pcm.data(), output_pcm.size(), num_channels,
                        sample_rate, 16);
  writer.Write(FLAGS_output_wav_path);
}
