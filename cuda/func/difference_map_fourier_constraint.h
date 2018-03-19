#pragma once

#include "utils/Complex.h"
#include "utils/CudaFunction.h"
#include "utils/Memory.h"

#include "difference_map_realspace_constraint.h"
#include "far_field_error.h"
#include "farfield_propagator.h"
#include "get_difference.h"
#include "log_likelihood.h"
#include "realspace_error.h"
#include "renormalise_fourier_magnitudes.h"
#include "scan_and_multiply.h"
#include "sum_to_buffer.h"

class DifferenceMapFourierConstraint : public CudaFunction
{
public:
  DifferenceMapFourierConstraint(int i,
                                 int m,
                                 int n,
                                 int obj_m,
                                 int obj_n,
                                 int probe_m,
                                 int probe_n,
                                 bool do_LL_error,
                                 bool do_realspace_error);
  void setDeviceBuffers(unsigned char *d_mask,
                        float *d_Idata,
                        complex<float> *d_obj,
                        complex<float> *d_probe,
                        complex<float> *d_exit_wave,
                        int *d_addr_info,
                        complex<float> *d_prefilter,
                        complex<float> *d_postfilter,
                        float *d_errors,
                        int *d_outidx,
                        int *d_startidx,
                        int *d_indices,
                        int outidx_size);
  int calculateAddrIndices(const int *out1_addr);
  void allocate();
  void transfer_in(const unsigned char *mask,
                   const float *Idata,
                   const complex<float> *obj,
                   const complex<float> *probe,
                   const complex<float> *exit_wave,
                   const int *addr_info,
                   const complex<float> *prefilter,
                   const complex<float> *postfilter);
  void run(float pbound, float alpha, bool doPbound);
  void transfer_out(float *errors, complex<float> *exit_wave);

private:
  DevicePtrWrapper<unsigned char> d_mask_;
  DevicePtrWrapper<float> d_Idata_;
  DevicePtrWrapper<complex<float>> d_obj_;
  DevicePtrWrapper<complex<float>> d_probe_;
  DevicePtrWrapper<complex<float>> d_exit_wave_;
  DevicePtrWrapper<int> d_addr_info_;
  DevicePtrWrapper<complex<float>> d_prefilter_;
  DevicePtrWrapper<complex<float>> d_postfilter_;
  DevicePtrWrapper<float> d_errors_;

  DevicePtrWrapper<int> d_outidx_, d_startidx_, d_indices_;
  std::vector<int> outidx_, startidx_, indices_;
  int outidx_size_;

  int i_, m_, n_;
  int obj_m_, obj_n_;
  int probe_m_, probe_n_;
  bool do_LL_error_;
  bool do_realspace_error_;
  // intermediate buffers
  DevicePtrWrapper<complex<float>> d_prefilter_conj_;
  DevicePtrWrapper<complex<float>> d_postfilter_conj_;
  DevicePtrWrapper<float> d_fmag_;
  // other kernels
  ScanAndMultiply scan_and_multiply_;
  LogLikelihood log_likelihood_;
  DifferenceMapRealspaceConstraint difference_map_realspace_constraint_;
  FarfieldPropagator farfield_propagator_fwd_;
  Abs2<complex<float>, float> abs2_;
  SumToBuffer<float> sum_to_buffer_;
  FarFieldError far_field_error_;
  RenormaliseFourierMagnitudes renormalise_fourier_magnitudes_;
  FarfieldPropagator farfield_propagator_rev_;
  GetDifference get_difference_;
  RealspaceError realspace_error_;
};
