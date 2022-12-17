#include <sstream>

#include "vad/onnx_model.h"

#include "glog/logging.h"

Ort::Env OnnxModel::env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "");
Ort::SessionOptions OnnxModel::session_options_ = Ort::SessionOptions();

void OnnxModel::InitEngineThreads(int num_threads) {
  session_options_.SetIntraOpNumThreads(num_threads);
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
}

static std::wstring ToWString(const std::string& str) {
  unsigned len = str.size() * 2;
  setlocale(LC_CTYPE, "");
  wchar_t* p = new wchar_t[len];
  mbstowcs(p, str.c_str(), len);
  std::wstring wstr(p);
  delete[] p;
  return wstr;
}

OnnxModel::OnnxModel(const std::string& model_path) {
  InitEngineThreads(1);
#ifdef _MSC_VER
  session_ = std::make_shared<Ort::Session>(env_, ToWString(model_path).c_str(),
                                            session_options_);
#else
  session_ = std::make_shared<Ort::Session>(env_, model_path.c_str(),
                                            session_options_);
#endif
  Ort::AllocatorWithDefaultOptions allocator;
  // Input info
  int num_nodes = session_->GetInputCount();
  input_node_names_.resize(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    input_node_names_[i] = session_->GetInputName(i, allocator);
    LOG(INFO) << "Input names[" << i << "]: " << input_node_names_[i];
  }
  // Output info
  num_nodes = session_->GetOutputCount();
  output_node_names_.resize(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    output_node_names_[i] = session_->GetOutputName(i, allocator);
    LOG(INFO) << "Output names[" << i << "]: " << output_node_names_[i];
  }
}
