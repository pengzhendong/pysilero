#include "vad/vad_model.h"

void VadModel::Reset() {
  _h.resize(size_hc);
  _c.resize(size_hc);
  sr.resize(1);
  std::memset(_h.data(), 0.0f, size_hc * sizeof(float));
  std::memset(_c.data(), 0.0f, size_hc * sizeof(float));
  triggerd = false;
  temp_end = 0;
  current_sample = 0;
}

void VadModel::Predict(const std::vector<float>& data) {
  int64_t input_node_dims[2] = {1, data.size()};
  Ort::Value input_ort = Ort::Value::CreateTensor<float>(
      memory_info_, const_cast<float*>(data.data()), data.size(),
      input_node_dims, 2);

  const int64_t sr_node_dims[1] = {1};
  Ort::Value sr_ort = Ort::Value::CreateTensor<int64_t>(memory_info_, sr.data(),
                                                        1, sr_node_dims, 1);
  const int64_t hc_node_dims[3] = {2, 1, 64};
  Ort::Value h_ort = Ort::Value::CreateTensor<float>(memory_info_, _h.data(),
                                                     size_hc, hc_node_dims, 3);
  Ort::Value c_ort = Ort::Value::CreateTensor<float>(memory_info_, _c.data(),
                                                     size_hc, hc_node_dims, 3);

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.emplace_back(std::move(input_ort));
  ort_inputs.emplace_back(std::move(sr_ort));
  ort_inputs.emplace_back(std::move(h_ort));
  ort_inputs.emplace_back(std::move(c_ort));

  // Infer
  auto ort_outputs = session_->Run(
      Ort::RunOptions{nullptr}, input_node_names_.data(), ort_inputs.data(),
      ort_inputs.size(), output_node_names_.data(), output_node_names_.size());

  // Output probability & update h, c recursively
  // MAX 4294967295 samples / 8 sample per ms / 1000 / 60 = 8947 minutes
  float output = ort_outputs[0].GetTensorMutableData<float>()[0];
  float* hn = ort_outputs[1].GetTensorMutableData<float>();
  float* cn = ort_outputs[2].GetTensorMutableData<float>();
  _h.assign(hn, hn + size_hc);
  _c.assign(cn, cn + size_hc);

  // Push forward sample index
  current_sample += data.size();

  // Reset temp_end when > threshold
  if (output >= threshold) {
    temp_end = 0;
  }
  // 1) Start
  if (output >= threshold && triggerd == false) {
    triggerd = true;
    // minus window_size_samples to get precise start time point.
    int speech_start = current_sample - data.size() - speech_pad_samples;
    printf("[%.3fs, ", 1.0 * speech_start / sample_rate);
  }
  // 2) End
  if (output < (threshold - 0.15) && triggerd == true) {
    if (temp_end != 0) {
      temp_end = current_sample;
    }
    if (current_sample - temp_end >= min_silence_samples) {
      int speech_end = current_sample + speech_pad_samples;
      temp_end = 0;
      triggerd = false;
      printf("%.3fs]\n", 1.0 * speech_end / sample_rate);
    }
  }
}