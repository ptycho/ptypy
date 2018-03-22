#pragma once

#include "utils/Complex.h"
#include "utils/CudaFunction.h"
#include "utils/Memory.h"

class RenormaliseFourierMagnitudes : public CudaFunction
{
public:
  RenormaliseFourierMagnitudes();
  void setParameters(int i, int m, int n);
  void setDeviceBuffers(complex<float> *d_f,
                        float *d_af,
                        float *d_fmag,
                        unsigned char *d_mask,
                        float *d_err_fmag,
                        int *d_addr_info,
                        complex<float> *d_out);
  void allocate();
  complex<float> *getOutput() const;
  void transfer_in(const complex<float> *f,
                   const float *af,
                   const float *fmag,
                   const unsigned char *mask,
                   const float *err_fmag,
                   const int *addr_info);
  void run(float pbound = 0.0, bool usePbound = false);
  void transfer_out(complex<float> *out);

private:
  DevicePtrWrapper<complex<float>> d_f_;
  DevicePtrWrapper<float> d_af_;
  DevicePtrWrapper<float> d_fmag_;
  DevicePtrWrapper<unsigned char> d_mask_;
  DevicePtrWrapper<float> d_err_fmag_;
  DevicePtrWrapper<int> d_addr_info_;
  DevicePtrWrapper<complex<float>> d_out_;
  int i_ = 0, m_ = 0, n_ = 0;
};