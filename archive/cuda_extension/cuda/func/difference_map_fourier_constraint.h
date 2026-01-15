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
  DifferenceMapFourierConstraint();
  void setParameters(int M,
                     int N,
                     int A,
                     int B,
                     int C,
                     int D,
                     int ob_modes,
                     int pr_modes,
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
  void calculateUniqueDaIndices(const int *da_addr);
  void allocate();
  // to be called if error point should be moved along
  // in engine iterator function
  void updateErrorOutput(float *d_errors);
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
  DevicePtrWrapper<unsigned char> d_mask_;         // N x A x B
  DevicePtrWrapper<float> d_Idata_;                // N x A x B
  DevicePtrWrapper<complex<float>> d_obj_;         // ob_modes x C x D
  DevicePtrWrapper<complex<float>> d_probe_;       // pr_modes x A x B
  DevicePtrWrapper<complex<float>> d_exit_wave_;   // M x A x B
  DevicePtrWrapper<int> d_addr_info_;              // M x 5 x 3
  DevicePtrWrapper<complex<float>> d_prefilter_;   // A x B
  DevicePtrWrapper<complex<float>> d_postfilter_;  // A x B
  DevicePtrWrapper<float> d_errors_;               // 3 x N

  DevicePtrWrapper<int> d_outidx_, d_startidx_, d_indices_;
  std::vector<int> outidx_, startidx_, indices_;
  int outidx_size_ = 0;

  int M_ = 0, N_ = 0, A_ = 0, B_ = 0, C_ = 0, D_ = 0, ob_modes_ = 0,
      pr_modes_ = 0;
  bool do_LL_error_ = 0;
  bool do_realspace_error_ = 0;
  // intermediate buffers
  DevicePtrWrapper<complex<float>> d_prefilter_conj_;
  DevicePtrWrapper<complex<float>> d_postfilter_conj_;
  DevicePtrWrapper<float> d_fmag_;
  // other kernels
  ScanAndMultiply *scan_and_multiply_ = nullptr;
  LogLikelihood *log_likelihood_ = nullptr;
  DifferenceMapRealspaceConstraint *difference_map_realspace_constraint_ =
      nullptr;
  FarfieldPropagator *farfield_propagator_fwd_ = nullptr;
  Abs2<complex<float>, float> *abs2_ = nullptr;
  SumToBuffer<float> *sum_to_buffer_ = nullptr;
  FarFieldError *far_field_error_ = nullptr;
  RenormaliseFourierMagnitudes *renormalise_fourier_magnitudes_ = nullptr;
  FarfieldPropagator *farfield_propagator_rev_ = nullptr;
  GetDifference *get_difference_ = nullptr;
  RealspaceError *realspace_error_ = nullptr;
};
