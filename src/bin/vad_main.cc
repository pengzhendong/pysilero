#include <vector>

#include "gflags/gflags.h"

#include "vad/vad_model.h"
#include "wav.h"

DEFINE_string(wav_path, "", "wav path");
DEFINE_string(model_path, "", "voice activity detection model path");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  wav::WavReader wav_reader(FLAGS_wav_path);
  std::vector<float> input_wav(wav_reader.num_samples());
  for (int i = 0; i < wav_reader.num_samples(); i++) {
    input_wav[i] = wav_reader.data()[i] / 32768;
  }

  int test_sr = 8000;
  float test_threshold = 0.5f;
  int test_min_silence_duration_ms = 0;
  int test_speech_pad_ms = 0;
  VadModel vad(FLAGS_model_path, test_sr, test_threshold,
               test_min_silence_duration_ms, test_speech_pad_ms);

  int test_frame_ms = 64;
  int test_window_samples = test_frame_ms * (test_sr / 1000);
  // Assign when init, support 256 512 768 for 8k; 512 1024 1536 for 16k.
  for (int j = 0; j < wav_reader.num_samples(); j += test_window_samples) {
    std::vector<float> r{&input_wav[0] + j,
                         &input_wav[0] + j + test_window_samples};
    vad.Predict(r);
  }
}