#pragma once

#include "utils/Complex.h"
#include "utils/CudaFunction.h"
#include "utils/Memory.h"

class RenormaliseFourierMagnitudes : public CudaFunction
{
public:
  RenormaliseFourierMagnitudes();
  void setParameters(int M, int N, int A, int B);
  void setDeviceBuffers(complex<float> *d_f,
                        float *d_af,
                        float *d_fmag,
                        unsigned char *d_mask,
                        float *d_err_fmag,
                        int *d_addr_info,
                        complex<float> *d_out);
  void allocate();
  void updateErrorInput(float* d_err_fmag);
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
  DevicePtrWrapper<complex<float>> d_f_;    // M x A x B
  DevicePtrWrapper<float>  d_af_;           // M x A x B
  DevicePtrWrapper<float> d_fmag_;          // N x A x B
  DevicePtrWrapper<unsigned char> d_mask_;  // N x A X B
  DevicePtrWrapper<float> d_err_fmag_;      // N
  DevicePtrWrapper<int> d_addr_info_;       // M x 5 x 3
  DevicePtrWrapper<complex<float>> d_out_;  // M x A x B
  int M_ = 0, N_ = 0, A_ = 0, B_= 0;
};