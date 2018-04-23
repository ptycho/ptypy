#pragma once

#include "utils/CudaFunction.h"
#include "utils/Memory.h"

template <class T>
class Norm2 : public CudaFunction
{
public:
  Norm2();
  void setParameters(int size);
  void setDeviceBuffers(T* d_input, float* d_output);
  void allocate();
  float* getOutput() const;
  void transfer_in(const T* input);
  void run();
  void transfer_out(float* output);

private:
  static const int BLOCK_SIZE = 1024;
  DevicePtrWrapper<T> d_input_;
  DevicePtrWrapper<float> d_intermediate_;
  DevicePtrWrapper<float> d_output_;
  int size_ = 0;
  int numblocks_ = 0;
};