# version 3.8+ is needed, otherwise there's no CUDA support
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(ptypy_cuda LANGUAGES CXX CUDA)
 
option(GPU_TIMING "Run timing for the GPU code" OFF)

include_directories(.)

add_library(gpu_extension STATIC
  utils/Complex.h
  utils/CudaFunction.h
  utils/CudaFunction.cpp
  utils/Errors.h
  utils/Memory.h
  utils/Memory.cpp
  utils/ScopedTimer.h
  utils/Timer.h
  utils/GpuManager.h 
  utils/GpuManager.cu
  
  func/addr_info_helpers.h
  func/addr_info_helpers.cpp

  func/farfield_propagator.h  
  func/farfield_propagator.cu 
  func/sqrt_abs.cu
  func/scan_and_multiply.h
  func/scan_and_multiply.cu 
  func/difference_map_realspace_constraint.h
  func/difference_map_realspace_constraint.cu
  func/log_likelihood.h
  func/log_likelihood.cu
  func/abs2.h
  func/abs2.cu
  func/sum_to_buffer.cu 
  func/sum_to_buffer.h
  func/far_field_error.h
  func/far_field_error.cu
  func/realspace_error.h
  func/realspace_error.cu
  func/get_difference.h
  func/get_difference.cu
  func/renormalise_fourier_magnitudes.h
  func/renormalise_fourier_magnitudes.cu
  func/difference_map_fourier_constraint.h
  func/difference_map_fourier_constraint.cu
  func/norm2.h
  func/norm2.cu
  func/mass_center.h
  func/mass_center.cu
  func/clip_complex_magnitudes_to_range.h
  func/clip_complex_magnitudes_to_range.cu
  func/extract_array_from_exit_wave.h
  func/extract_array_from_exit_wave.cu
  func/interpolated_shift.cu
  func/interpolated_shift.h
)

target_compile_features(gpu_extension PUBLIC cxx_std_11)

set_target_properties(gpu_extension PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF  # no device-link symbols
  CUDA_SEPARABLE_COMPILATION OFF   # no device-side linking
  POSITION_INDEPENDENT_CODE ON     # allow linking into a dynamic lib
)

# add "DO_GPU_TIMING" if option is true
if(GPU_TIMING)
  set_target_properties(gpu_extension PROPERTIES 
    COMPILE_DEFINITIONS DO_GPU_TIMING
  )
endif()

