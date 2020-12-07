#pragma once

#include "utils/Complex.h"
#include "utils/CudaFunction.h"
#include "utils/Memory.h"

#include "complex_gaussian_filter.h"
#include "extract_array_from_exit_wave.h"
#include "clip_complex_magnitudes_to_range.h"

class DifferenceMapUpdateObject : public CudaFunction
{
public:
  DifferenceMapUpdateObject();
  void setParameters(int A,
                     int B,
                     int C,
                     int D,
                     int E,
                     int F,
                     int G,
                     int H,
                     int I,
                     float obj_smooth_std,
                     bool doSmoothing,
                     bool doClipping);
  void setDeviceBuffers(complex<float>* d_obj,
                        float* d_object_weights,
                        complex<float>* d_probe,
                        complex<float>* d_exit_wave,
                        int* d_addr_info,
                        complex<float>* d_cfact);
  void allocate();
  void transfer_in(const complex<float>* obj,
                   const float* object_weigths,
                   const complex<float>* probe,
                   const complex<float>* exit_wave,
                   const int* addr_info,
                   const complex<float>* cfact);
  void run(float clip_min, float clip_max);
  void transfer_out(complex<float>* obj);

private:
  DevicePtrWrapper<complex<float>> d_obj_;        // G x H x I
  DevicePtrWrapper<float> d_object_weights_;      // G
  DevicePtrWrapper<complex<float>> d_probe_;      // D x E x F
  DevicePtrWrapper<complex<float>> d_exit_wave_;  // A x B x C
  DevicePtrWrapper<int> d_addr_info_;             // A x 5 x 3
  DevicePtrWrapper<complex<float>> d_cfact_;      // G x H x I
  bool doSmoothing_ = false, doClipping_ = false;
  int A_ = 0, B_ = 0, C_ = 0, D_ = 0, E_ = 0, F_ = 0, G_ = 0, H_ = 0, I_ = 0;

  // child kernels
  ExtractArrayFromExitWave* extract_array_from_exit_wave_ = nullptr;
  ClipComplexMagnitudesToRange* clip_complex_magnitudes_to_range_ = nullptr;
  ComplexGaussianFilter* gaussian_filter_ = nullptr;
};