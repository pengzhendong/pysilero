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
  std::vector<float> input_wav{pcm, pcm + num_samples};

  bool denoise = true;
  float min_sil_dur = 0;  // in seconds
  float speech_pad = 0;   // in seconds
  VadModel vad(FLAGS_model_path, denoise, sample_rate, FLAGS_threshold,
               min_sil_dur, speech_pad);

  std::vector<float> start_pos;
  std::vector<float> stop_pos;
  float dur = vad.Vad(input_wav, &start_pos, &stop_pos);

  if (!stop_pos.empty() && stop_pos.back() > dur) {
    stop_pos.back() = dur;
  }
  if (stop_pos.size() < start_pos.size()) {
    stop_pos.emplace_back(dur);
  }
  for (int i = 0; i < start_pos.size(); i++) {
    LOG(INFO) << "[" << start_pos[i] << ", " << stop_pos[i] << "]s";
  }
}
