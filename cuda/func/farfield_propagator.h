#pragma once

#include "utils/Complex.h"
#include "utils/CudaFunction.h"
#include "utils/Memory.h"

#include <cufft.h>

class FarfieldPropagator : public CudaFunction
{
public:
  FarfieldPropagator();
  void setParameters(size_t batch_size, size_t m, size_t n);

  // for setting external memory to be used
  // (can be null if internal should be used)
  void setDeviceBuffers(complex<float> *d_datain,
                        complex<float> *d_dataout,
                        complex<float> *d_prefilter,
                        complex<float> *d_postfilter);

  void allocate();
  void transfer_in(const complex<float> *data_to_be_transformed,
                   const complex<float> *prefilter,
                   const complex<float> *postfilter);
  void transfer_out(complex<float> *out);
  void run(bool doPreFilter, bool doPostFilter, bool isForward);

  ~FarfieldPropagator();

private:
  size_t batch_size_ = 0, m_ = 0, n_ = 0;
  float sc_ = 1.0f;
  DevicePtrWrapper<complex<float>> d_datain_;
  DevicePtrWrapper<complex<float>> d_dataout_;
  DevicePtrWrapper<complex<float>> d_pre_;
  DevicePtrWrapper<complex<float>> d_post_;
  cufftHandle plan_ = 0;
  // to keep track of previously-used plan to avoid realloc
  size_t old_batch_size_ = 0, old_m_ = 0, old_n_ = 0;
};
