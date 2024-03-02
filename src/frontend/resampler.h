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

#ifndef FRONTEND_RESAMPLER_H_
#define FRONTEND_RESAMPLER_H_

#include <memory>
#include <vector>

#include "glog/logging.h"
#include "samplerate.h"

class Resampler {
 public:
  explicit Resampler(int in_sr, int out_sr,
                     int converter = SRC_SINC_BEST_QUALITY);
  ~Resampler() { src_delete(src_state_); }

  void Reset() { src_reset(src_state_); }

  void Resample(const std::vector<float>& in_pcm, std::vector<float>* out_pcm,
                int enf_of_input = 0);

 private:
  float src_ratio_;
  SRC_STATE* src_state_;
};

#endif  // FRONTEND_RESAMPLER_H_
