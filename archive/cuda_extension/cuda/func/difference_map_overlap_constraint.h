#pragma once
#include "utils/Complex.h"
#include "utils/CudaFunction.h"
#include "utils/Memory.h"

#include "center_probe.h"
#include "difference_map_update_object.h"
#include "difference_map_update_probe.h"

class DifferenceMapOverlapConstraint : public CudaFunction
{
public:
  DifferenceMapOverlapConstraint();
  void setParameters(
      int A,
      int B,
      int C,
      int D,
      int E,
      int F,
      int G,
      int H,
      int I,
      float obj_smooth_std,
      bool doUpdateObjectFirst,
      bool doUpdateProbe,  // in general (alloc space / ops for it)
      bool doSmoothing,
      bool doClipping,
      bool withProbeSupport,
      bool doCentering);
  void setDeviceBuffers(int* d_addr_info,
                        complex<float>* d_cfact_object,
                        complex<float>* d_cfact_probe,
                        complex<float>* d_exit_wave,
                        complex<float>* d_obj,
                        float* d_obj_weigths,
                        complex<float>* d_probe,
                        complex<float>* d_probe_support,
                        float* d_probe_weights);
  void allocate();
  void transfer_in(const int* addr_info,
                   const complex<float>* cfact_object,
                   const complex<float>* cfact_probe,
                   const complex<float>* exit_wave,
                   const complex<float>* obj,
                   const float* obj_weigths,
                   const complex<float>* probe,
                   const complex<float>* probe_support,
                   const float* probe_weights);
  void run(
      int max_iterations,
      float clip_min,
      float clip_max,
      float probe_center_tol,
      float overlap_converge_factor,
      bool do_update_probe = true  // update it in this call? (logical and with
                                   // the general setParameters one is done)
  );
  void transfer_out(complex<float>* probe, complex<float>* obj);

private:
  DevicePtrWrapper<int> d_addr_info_;                 // A x 5 x 3
  DevicePtrWrapper<complex<float>> d_cfact_object_;   // G x H x I
  DevicePtrWrapper<complex<float>> d_cfact_probe_;    // D x E x F
  DevicePtrWrapper<complex<float>> d_exit_wave_;      // A x B x C
  DevicePtrWrapper<complex<float>> d_obj_;            // G x H x I
  DevicePtrWrapper<float> d_obj_weights_;             // G
  DevicePtrWrapper<complex<float>> d_probe_;          // D x E x F
  DevicePtrWrapper<complex<float>> d_probe_support_;  // D x E x F
  DevicePtrWrapper<float> d_probe_weights_;           // D
  bool doUpdateObjectFirst_ = false;
  bool doUpdateProbe_ = true;
  bool doSmoothing_ = false;
  bool doClipping_ = true;
  bool withProbeSupport_ = true;
  bool doCentering_ = true;
  int A_ = 0, B_ = 0, C_ = 0, D_ = 0, E_ = 0, F_ = 0, G_ = 0, H_ = 0, I_ = 0;

  DifferenceMapUpdateProbe* dm_update_probe_ = nullptr;
  DifferenceMapUpdateObject* dm_update_object_ = nullptr;
  CenterProbe* center_probe_ = nullptr;
};