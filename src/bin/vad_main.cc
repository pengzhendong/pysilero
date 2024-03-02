// Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
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

#include <iostream>
#include <vector>

#include "gflags/gflags.h"

#include "frontend/wav.h"
#include "vad/vad_model.h"

DEFINE_string(wav_path, "", "wav path");
DEFINE_double(threshold, 0.5, "threshold of voice activity detection");
DEFINE_string(model_path, "", "voice activity detection model path");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  wav::WavReader wav_reader(FLAGS_wav_path);
  int num_channels = wav_reader.num_channels();
  CHECK_EQ(num_channels, 1) << "Only support mono (1 channel) wav!";
  int sample_rate = wav_reader.sample_rate();
  const float* pcm = wav_reader.data();
  int num_samples = wav_reader.num_samples();

  const int frame_size_ms = 10;
  const int frame_size_samples = frame_size_ms * sample_rate / 1000;
  VadModel vad(FLAGS_model_path, true, sample_rate, FLAGS_threshold);

  for (int i = 0; i < num_samples; i += frame_size_samples) {
    // Extract 10ms frame from input_pcm
    int remaining_samples = std::min(frame_size_samples, num_samples - i);
    std::vector<float> input_pcm(pcm + i, pcm + i + remaining_samples);
    vad.AcceptWaveform(input_pcm);
    float speech_start = -1;
    float speech_end = -1;
    vad.Vad(&speech_start, &speech_end, false, true);
    if (speech_start >= 0) {
      std::cout << "[" << speech_start << ", ";
    }
    if (speech_end >= 0) {
      std::cout << speech_end << "]" << std::endl;
    }
  }
}
