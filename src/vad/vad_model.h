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
           float threshold, int min_sil_dur_ms = 100, int speech_pad_ms = 30);

  void Reset();

  void AcceptWaveform(const std::vector<float>& pcm);
  void Vad(const std::vector<float>& pcm, std::vector<float>* start_pos,
           std::vector<float>* end_pos);

 private:
  float Forward(const std::vector<float>& pcm);

  // 16k sample rate:
  // - frame_samples: 512 1024 1536
  // - frame_ms: 32 64 96
  int frame_ms_ = 32;
  int frame_size_ = frame_ms_ * (16000 / 1000);

  bool denoise_ = false;
  int sample_rate_;
  float threshold_;
  int min_sil_dur_samples_;
  int speech_pad_samples_;

  // model states
  bool on_speech_ = false;
  float temp_end_ = 0;
  int current_sample_ = 0;

  // Onnx model
  std::vector<float> h_;
  std::vector<float> c_;

  std::shared_ptr<Denoiser> denoiser_ = nullptr;
  std::shared_ptr<Resampler> resampler_ = nullptr;
  std::shared_ptr<SampleQueue> sample_queue_ = nullptr;
};

#endif  // VAD_VAD_MODEL_H_
