#pragma once
#include "utils/Complex.h"
#include "utils/CudaFunction.h"
#include "utils/Memory.h"

#include "func/abs2.h"
#include "func/farfield_propagator.h"
#include "func/sum_to_buffer.h"

class LogLikelihood : public CudaFunction
{
public:
  LogLikelihood();
  void setParameters(int i, int m, int n, int addr_i, int Idata_i);

  void setDeviceBuffers(complex<float> *d_probe_obj,
                        unsigned char *d_mask,
                        float *d_Idata,
                        complex<float> *d_prefilter,
                        complex<float> *d_postfilter,
                        int *d_addr_info,
                        float *d_out,
                        int *d_outidx,
                        int *d_startidx,
                        int *d_indices,
                        int outidx_size);
  int calculateAddrIndices(const int *out1_addr);
  void calculateUniqueDaIndices(const int *da_addr);
  void allocate();
  void updateErrorOutput(float *d_out);
  float *getOutput() const;
  void transfer_in(const complex<float> *probe_obj,
                   const unsigned char *mask,
                   const float *Idata,
                   const complex<float> *prefilter,
                   const complex<float> *postfilter,
                   const int *addr_info);
  void run();
  void transfer_out(float *out);

private:
  DevicePtrWrapper<complex<float>> d_probe_obj_;
  DevicePtrWrapper<unsigned char> d_mask_;
  DevicePtrWrapper<float> d_Idata_;
  DevicePtrWrapper<complex<float>> d_prefilter_;
  DevicePtrWrapper<complex<float>> d_postfilter_;
  DevicePtrWrapper<int> d_addr_info_;
  DevicePtrWrapper<float> d_out_;
  DevicePtrWrapper<float> d_LL_;
  // internal buffer for intermediate results
  DevicePtrWrapper<complex<float>> d_ft_;
  DevicePtrWrapper<float> d_abs2_ft_;
  // these three are for bookkeeping between setDeviceBuffers and allocate
  // so that they can be forwarded to sum2buffer
  int *d_outidx_ = nullptr;
  int *d_startidx_ = nullptr;
  int *d_indices_ = nullptr;
  int outidx_size_ = 0;
  // unique indices for da
  DevicePtrWrapper<int> d_da_unique_;

  int i_ = 0, m_ = 0, n_ = 0, addr_i_ = 0;
  int Idata_i_ = 0;
  FarfieldPropagator *ffprop_ = nullptr;
  Abs2<complex<float>, float> *abs2_ = nullptr;
  SumToBuffer<float> *sum2buffer_ = nullptr;
};
