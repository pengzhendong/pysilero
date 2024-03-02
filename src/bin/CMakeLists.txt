add_executable(denoise_main denoise_main.cc)
target_link_libraries(denoise_main PUBLIC frontend gflags)

add_executable(resample_main resample_main.cc)
target_link_libraries(resample_main PUBLIC frontend gflags)

add_executable(vad_main vad_main.cc)
target_link_libraries(vad_main PUBLIC vad gflags)
