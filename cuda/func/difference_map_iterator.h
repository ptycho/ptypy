#pragma once
#include "utils/Complex.h"
#include "utils/CudaFunction.h"
#include "utils/Memory.h"

#include "func/difference_map_fourier_constraint.h"
#include "func/difference_map_overlap_constraint.h"

class DifferenceMapIterator : public CudaFunction
{
public:
  DifferenceMapIterator();
  void setParameters(int A,
                     int B,
                     int C,
                     int D,
                     int E,
                     int F,
                     int G, 
                     int H, 
                     int I,
                     int N,
                     int num_iterations,
                     float obj_smooth_std,
                     bool doSmoothing,
                     bool doClipping,
                     bool doCentering,
                     bool doPbound,
                     bool do_LL_error,
                     bool doRealspaceError,
                     bool doUpdateObjectFirst,
                     bool doProbeSupport);
  void setDeviceBuffers(float* d_diffraction,
                        complex<float>* d_obj,
                        float* d_object_weights,
                        complex<float>* d_cfact_object,
                        unsigned char* d_mask,
                        complex<float>* d_probe,
                        complex<float>* d_cfact_probe,
                        complex<float>* d_probe_support,
                        float* d_probe_weight,
                        complex<float>* d_exit_wave,
                        int* d_addr_info,
                        complex<float>* d_pre_fft,
                        complex<float>* d_post_fft,
                        float* d_errors);
  int calculateAddrIndices(const int* out1_addr);
  void calculateUniqueDaIndices(const int* da_addr);
  void allocate();
  void transfer_in(const float* diffraction,
                   const complex<float>* obj,
                   const float* object_weights,
                   const complex<float>* cfact_object,
                   const unsigned char* mask,
                   const complex<float>* probe,
                   const complex<float>* cfact_probe,
                   const complex<float>* probe_support,
                   const float* probe_weight,
                   const complex<float>* exit_wave,
                   const int* addr_info,
                   const complex<float>* pre_fft,
                   const complex<float>* post_fft);
  void run(int overlap_max_iterations,
           float overlap_converge_factor,
           float probe_center_tol,
           int probe_update_start,
           float pbound,
           float alpha,
           float clip_min,
           float clip_max);
  void transfer_out(float* errors,
                    complex<float>* obj,
                    complex<float>* probe,
                    complex<float>* exit_wave);

private:
  DevicePtrWrapper<float> d_diffraction_;    // N x B x C
  DevicePtrWrapper<complex<float>> d_obj_;            // G x H x I
  DevicePtrWrapper<float> d_object_weights_;          // G
  DevicePtrWrapper<complex<float>> d_cfact_object_;   // G x H x I
  DevicePtrWrapper<unsigned char> d_mask_;            // N x B x C
  DevicePtrWrapper<complex<float>> d_probe_;          // D x E x F
  DevicePtrWrapper<complex<float>> d_cfact_probe_;    // D x E x F
  DevicePtrWrapper<complex<float>> d_probe_support_;  // D x E x F
  DevicePtrWrapper<float> d_probe_weights_;           // D
  DevicePtrWrapper<complex<float>> d_exit_wave_;      // A x B x C
  DevicePtrWrapper<int> d_addr_info_;                 // A x 5 x 3
  DevicePtrWrapper<complex<float>> d_pre_fft_;        // B x C
  DevicePtrWrapper<complex<float>> d_post_fft_;       // B x C
  DevicePtrWrapper<float> d_errors_;                  // num_iterations x 3 x N

  DevicePtrWrapper<int> d_outidx_, d_startidx_, d_indices_;
  std::vector<int> outidx_, startidx_, indices_;
  int outidx_size_ = 0;

  int A_ = 0, B_ = 0, C_ = 0, D_ = 0, E_ = 0, F_ = 0, G_ = 0, H_ = 0, I_ = 0,
      N_ = 0;
  int num_iterations_ = 0;
  bool doPbound_ = false;
  bool doProbeSupport_ = false;

  DifferenceMapFourierConstraint* dm_fourier_constraint_ = nullptr;
  DifferenceMapOverlapConstraint* dm_overlap_update_ = nullptr;
};