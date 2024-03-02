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

#include "frontend/resampler.h"

Resampler::Resampler(int in_sr, int out_sr, int converter) {
  src_state_ = src_new(converter, 1, nullptr);
  src_ratio_ = out_sr * 1.0 / in_sr;
  src_set_ratio(src_state_, src_ratio_);
}

void Resampler::Resample(const std::vector<float>& in_pcm,
                         std::vector<float>* out_pcm, int end_of_input) {
  out_pcm->resize(in_pcm.size() * src_ratio_);

  SRC_DATA src_data;
  src_data.src_ratio = src_ratio_;
  src_data.end_of_input = end_of_input;
  src_data.data_in = in_pcm.data();
  src_data.input_frames = in_pcm.size();
  src_data.data_out = out_pcm->data();
  src_data.output_frames = out_pcm->size();

  int error = src_process(src_state_, &src_data);
  if (error != 0) {
    LOG(FATAL) << "src_process error: " << src_strerror(error);
  }
}
