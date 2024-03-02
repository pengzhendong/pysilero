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

#include <signal.h>

#include <iostream>
#include <vector>

#include "gflags/gflags.h"
#include "portaudio.h"

#include "vad/vad_model.h"

DEFINE_double(threshold, 0.5, "threshold of voice activity detection");
DEFINE_string(model_path, "", "voice activity detection model path");

int g_exiting = 0;
int sample_rate = 16000;
std::shared_ptr<VadModel> vad;

void SigRoutine(int dunno) {
  if (dunno == SIGINT) {
    g_exiting = 1;
  }
}

static int RecordCallback(const void* input, void* output,
                          unsigned long frames_count,
                          const PaStreamCallbackTimeInfo* time_info,
                          PaStreamCallbackFlags status_flags, void* user_data) {
  const auto* pcm_data = static_cast<const int16_t*>(input);
  std::vector<float> pcm(pcm_data, pcm_data + frames_count);
  vad->AcceptWaveform(pcm);
  if (g_exiting) {
    LOG(INFO) << "Exiting loop.";
    return paComplete;
  } else {
    return paContinue;
  }
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);
  vad = std::make_shared<VadModel>(FLAGS_model_path, true, sample_rate,
                                   FLAGS_threshold);

  signal(SIGINT, SigRoutine);
  PaError err = Pa_Initialize();
  PaStreamParameters params;
  std::cout << err << " " << Pa_GetDeviceCount() << std::endl;
  params.device = Pa_GetDefaultInputDevice();
  if (params.device == paNoDevice) {
    LOG(FATAL) << "Error: No default input device.";
  }
  params.channelCount = 1;
  params.sampleFormat = paInt16;
  params.suggestedLatency =
      Pa_GetDeviceInfo(params.device)->defaultLowInputLatency;
  params.hostApiSpecificStreamInfo = NULL;
  PaStream* stream;
  // Callback and process pcm date each `interval` ms.
  int interval_ms = 10;
  const int frame_size_samples = interval_ms * sample_rate / 1000;
  Pa_OpenStream(&stream, &params, NULL, sample_rate, frame_size_samples,
                paClipOff, RecordCallback, NULL);
  Pa_StartStream(stream);
  LOG(INFO) << "=== Now recording!! Please speak into the microphone. ===";

  while (Pa_IsStreamActive(stream)) {
    float speech_start = -1;
    float speech_end = -1;
    vad->Vad(&speech_start, &speech_end, false, true);
    if (speech_start >= 0) {
      LOG(INFO) << "start: " << speech_start;
    }
    if (speech_end >= 0) {
      LOG(INFO) << "end: " << speech_end;
    }
  }
  Pa_StopStream(stream);
  Pa_CloseStream(stream);
  Pa_Terminate();
}
