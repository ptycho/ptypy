#pragma once
#include "utils/CudaFunction.h"
#include "utils/Memory.h"

class FarFieldError : public CudaFunction
{
public:
  FarFieldError();
  void setParameters(int i, int m, int n);
  void setDeviceBuffers(float *d_current,
                        float *d_measured,
                        unsigned char *d_mask,
                        float *d_out);
  void allocate();
  void updateErrorOutput(float *d_out);
  float *getOutput() const;
  void transfer_in(const float *current,
                   const float *measured,
                   const unsigned char *mask);
  void run();
  void transfer_out(float *out);

private:
  int i_ = 0, m_ = 0, n_ = 0;
  DevicePtrWrapper<unsigned char> d_mask_;
  DevicePtrWrapper<float> d_current_;
  DevicePtrWrapper<float> d_measured_;
  DevicePtrWrapper<float> d_out_;
};
