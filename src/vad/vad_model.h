#ifndef VAD_VAD_MODEL_H_
#define VAD_VAD_MODEL_H_

#include "vad/onnx_model.h"

#include <memory>

#include "frontend/denoiser.h"
#include "frontend/resampler.h"
#include "frontend/sample_queue.h"

#define SIZE_HC 128  // 2 * 1 * 64

class VadModel : public OnnxModel {
 public:
  VadModel(const std::string& model_path, bool denoise, int sample_rate,
           float threshold, float min_sil_dur, float speech_pad);

  void Reset();

  float Vad(const std::vector<float>& pcm, std::vector<float>* start_pos,
            std::vector<float>* stop_pos);

 private:
  float Forward(const std::vector<float>& pcm);

  bool denoise_ = false;
  int sample_rate_;
  float threshold_;
  float min_sil_dur_;
  float speech_pad_;

  // model states
  bool on_speech_ = false;
  float temp_stop_ = 0;
  float current_pos_ = 0;

  // Onnx model
  std::vector<float> h_;
  std::vector<float> c_;

  std::shared_ptr<Denoiser> denoiser_ = nullptr;
  std::shared_ptr<Resampler> resampler_ = nullptr;
  std::shared_ptr<SampleQueue> sample_queue_ = nullptr;
};

#endif  // VAD_VAD_MODEL_H_
