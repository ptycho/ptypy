#pragma once

#include "utils/Complex.h"
#include "utils/CudaFunction.h"
#include "utils/Memory.h"

#include "extract_array_from_exit_wave.h"
#include "norm2.h"

class DifferenceMapUpdateProbe : public CudaFunction
{
public:
  DifferenceMapUpdateProbe();
  void setParameters(int A,
                     int B,
                     int C,
                     int D,
                     int E,
                     int F,
                     int G,
                     int H,
                     int I,
                     bool withProbeSupport);
  void setDeviceBuffers(complex<float>* d_obj,
                        float* d_probe_weights,
                        complex<float>* d_probe,
                        complex<float>* d_exit_wave,
                        int* d_addr_info,
                        complex<float>* d_cfact_probe,
                        complex<float>* d_probe_support);
  void allocate();
  void transfer_in(const complex<float>* obj,
                   const float* probe_weights,
                   const complex<float>* probe,
                   const complex<float>* exit_wave,
                   const int* addr_info,
                   const complex<float>* cfact_probe,
                   const complex<float>* probe_support);
  void run();
  void transfer_out(complex<float>* probe, float* change);

private:
  DevicePtrWrapper<complex<float>> d_obj_;
  DevicePtrWrapper<float> d_probe_weights_;
  DevicePtrWrapper<complex<float>> d_probe_;
  DevicePtrWrapper<complex<float>> d_exit_wave_;
  DevicePtrWrapper<int> d_addr_info_;
  DevicePtrWrapper<complex<float>> d_cfact_probe_;
  DevicePtrWrapper<complex<float>> d_probe_support_;
  int A_ = 0, B_ = 0, C_ = 0, D_ = 0, E_ = 0, F_ = 0, G_ = 0, H_ = 0, I_ = 0;
  bool withProbeSupport_ = true;

  // temporary buffers
  DevicePtrWrapper<complex<float>> d_buffer_; // size = probe

  // child kernels
  ExtractArrayFromExitWave* extract_array_from_exit_wave_ = nullptr;
  Norm2<complex<float>>* norm2_probe_;
  Norm2<complex<float>>* norm2_diff_;
};