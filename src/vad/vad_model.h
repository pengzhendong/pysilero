#ifndef VAD_VAD_MODEL_H_
#define VAD_VAD_MODEL_H_

#include "vad/onnx_model.h"

class VadModel : public OnnxModel {
 public:
  VadModel(const std::string& model_path, int sample_rate, float threshold,
           int min_silence_duration_ms, int speech_pad_ms)
      : OnnxModel(model_path), sample_rate(sample_rate), threshold(threshold) {
    int sr_per_ms = sample_rate / 1000;  // Assign when init, support 8 or 16
    min_silence_samples = sr_per_ms * min_silence_duration_ms;
    speech_pad_samples = sr_per_ms * speech_pad_ms;
    Reset();
  }

  void Reset();
  void Predict(const std::vector<float>& data);

 private:
  int sample_rate;
  float threshold;
  int min_silence_samples;  // sr_per_ms * #ms
  int speech_pad_samples;   // usually a

  // model states
  bool triggerd = false;
  unsigned int temp_end = 0;
  unsigned int current_sample = 0;

  // Onnx model
  int size_hc = 2 * 1 * 64;  // It's FIXED.
  std::vector<float> _h;
  std::vector<float> _c;
  std::vector<int64_t> sr;
};

#endif  // VAD_VAD_MODEL_H_
