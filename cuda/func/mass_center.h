#pragma once
#include "utils/CudaFunction.h"
#include "utils/Memory.h"

class MassCenter : public CudaFunction
{
public:
  MassCenter();
  void setParameters(int i, int m, int n = 1);
  void setDeviceBuffers(float* d_data, float* d_out);
  void allocate();
  void transfer_in(const float* data);
  void run();
  void transfer_out(float* out);

private:
  DevicePtrWrapper<float> d_data_;
  DevicePtrWrapper<float> d_i_sum_, d_m_sum_, d_n_sum_;
  DevicePtrWrapper<float> d_out_;
  int i_ = 0, m_ = 0, n_ = 0;
};