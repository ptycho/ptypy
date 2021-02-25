#pragma once
#include "utils/Complex.h"
#include "utils/CudaFunction.h"
#include "utils/Memory.h"

class RealspaceError : public CudaFunction
{
public:
  RealspaceError();
  void setParameters(int i, int m, int n, int addr_len, int outlen);
  void setAddrStride(int addr_stride);
  void setDeviceBuffers(complex<float>* d_difference,
                        int* d_ea_first_column,
                        int* d_da_first_column,
                        float* d_out);
  void allocate();
  void updateErrorOutput(float* d_out);
  void transfer_in(const complex<float>* difference,
                   const int* ea_first_column,
                   const int* da_first_column);
  void run();
  void transfer_out(float* out);

private:
  DevicePtrWrapper<complex<float>> d_difference_;
  DevicePtrWrapper<int> d_ea_first_column_;
  DevicePtrWrapper<int> d_da_first_column_;
  DevicePtrWrapper<float> d_out_;
  int i_ = 0, m_ = 0, n_ = 0, addr_len_ = 0, outlen_ = 0;
  int addr_stride_ = 1;
};
