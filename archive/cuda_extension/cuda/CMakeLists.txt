# version 3.8+ is needed, otherwise there's no CUDA support
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(ptypy_cuda LANGUAGES CXX CUDA)
 
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

option(GPU_TIMING "Run timing for the GPU code" OFF)

############################################################################
## Download and unpack googletest at configure time
#configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
#execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
#  RESULT_VARIABLE result
#  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
#if(result)
#  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
#endif()
#execute_process(COMMAND ${CMAKE_COMMAND} --build .
#  RESULT_VARIABLE result
#  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
#if(result)
#  message(FATAL_ERROR "Build step for googletest failed: ${result}")
#endif()

## Prevent overriding the parent project's compiler/linker
## settings on Windows
#set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
#
## Add googletest directly to our build. This defines
## the gtest and gtest_main targets.
#add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
#                 ${CMAKE_BINARY_DIR}/googletest-build
#                 EXCLUDE_FROM_ALL)
#
## The gtest/gtest_main targets carry header search path
## dependencies automatically when using CMake 2.8.11 or
## later. Otherwise we have to add them here ourselves.
#if (CMAKE_VERSION VERSION_LESS 2.8.11)
#  include_directories("${gtest_SOURCE_DIR}/include")
#endif()
############################################################################


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
  func/center_probe.h
  func/center_probe.cu
  func/difference_map_update_probe.h
  func/difference_map_update_probe.cu
  func/difference_map_update_object.h
  func/difference_map_update_object.cu
  func/difference_map_overlap_constraint.h
  func/difference_map_overlap_constraint.cu
  func/complex_gaussian_filter.h
  func/complex_gaussian_filter.cu
  func/difference_map_iterator.h
  func/difference_map_iterator.cu
)

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

########### tests ###########

#enable_testing()

#macro(buildtest name)
#  add_executable(${name} tests/${name}.cpp)
#  target_link_libraries(${name} gtest_main)
#  add_test(NAME ${name} COMMAND ${name})
#endmacro()

#buildtest(indexing_test)
#buildtest(gaussian_weights_test)

