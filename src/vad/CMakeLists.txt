add_library(vad STATIC onnx_model.cc vad_model.cc)
target_link_libraries(vad PUBLIC frontend glog onnxruntime)
