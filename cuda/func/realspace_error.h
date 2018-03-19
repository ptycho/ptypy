#pragma once
#include "utils/Complex.h"
#include "utils/CudaFunction.h"
#include "utils/Memory.h"

class RealspaceError : public CudaFunction
{
public:
  RealspaceError(int i, int m, int n);
  void setDeviceBuffers(complex<float>* d_difference, float* d_out);
  void allocate();
  void transfer_in(const complex<float>* difference);
  void run();
  void transfer_out(float* out);

private:
  DevicePtrWrapper<complex<float>> d_difference_;
  DevicePtrWrapper<float> d_out_;
  int i_, m_, n_;
};
