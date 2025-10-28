#include "difference_map_overlap_constraint.h"

#include "utils/Complex.h"
#include "utils/GpuManager.h"
#include "utils/ScopedTimer.h"

/********* class implementation ************/

DifferenceMapOverlapConstraint::DifferenceMapOverlapConstraint()
    : CudaFunction("difference_map_overlap_constraint")
{
}

void DifferenceMapOverlapConstraint::setParameters(int A,
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
                                                   bool doUpdateProbe,
                                                   bool doSmoothing,
                                                   bool doClipping,
                                                   bool withProbeSupport,
                                                   bool doCentering)
{
  A_ = A;
  B_ = B;
  C_ = C;
  D_ = D;
  E_ = E;
  F_ = F;
  G_ = G;
  H_ = H;
  I_ = I;
  doUpdateObjectFirst_ = doUpdateObjectFirst;
  doUpdateProbe_ = doUpdateProbe;
  doSmoothing_ = doSmoothing;
  doClipping_ = doClipping;
  withProbeSupport_ = withProbeSupport;
  doCentering_ = doCentering;

  dm_update_object_ = gpuManager.get_cuda_function<DifferenceMapUpdateObject>(
      "dm_overlap_constraint.dm_update_object",
      A,
      B,
      C,
      D,
      E,
      F,
      G,
      H,
      I,
      obj_smooth_std,
      doSmoothing,
      doClipping);
  if (doUpdateProbe_)
  {
    /* size parameter translation:
      D=>G,E=>H,F=>I,
      G=>D,H=>E,I=>F
    */
    dm_update_probe_ = gpuManager.get_cuda_function<DifferenceMapUpdateProbe>(
        "dm_overlap_constraint.dm_update_probe",
        A,
        B,
        C,
        G,
        H,
        I,
        D,
        E,
        F,
        withProbeSupport);
  }

  if (doCentering_)
  {
    center_probe_ = gpuManager.get_cuda_function<CenterProbe>(
        "dm_overlap_constraint.center_probe", D_, E_, F_);
  }
}

void DifferenceMapOverlapConstraint::setDeviceBuffers(
    int* d_addr_info,
    complex<float>* d_cfact_object,
    complex<float>* d_cfact_probe,
    complex<float>* d_exit_wave,
    complex<float>* d_obj,
    float* d_obj_weigths,
    complex<float>* d_probe,
    complex<float>* d_probe_support,
    float* d_probe_weights)
{
  d_addr_info_ = d_addr_info;
  d_cfact_object_ = d_cfact_object;
  d_cfact_probe_ = d_cfact_probe;
  d_exit_wave_ = d_exit_wave;
  d_obj_ = d_obj;
  d_obj_weights_ = d_obj_weigths;
  d_probe_ = d_probe;
  d_probe_support_ = d_probe_support;
  d_probe_weights_ = d_probe_weights;
}

void DifferenceMapOverlapConstraint::allocate()
{
  ScopedTimer t(this, "allocate");

  d_addr_info_.allocate(A_ * 15);
  d_cfact_object_.allocate(G_ * H_ * I_);
  d_cfact_probe_.allocate(D_ * E_ * F_);
  d_exit_wave_.allocate(A_ * B_ * C_);
  d_obj_.allocate(G_ * H_ * I_);
  d_obj_weights_.allocate(G_);
  d_probe_.allocate(D_ * E_ * F_);
  if (withProbeSupport_)
  {
    d_probe_support_.allocate(D_ * E_ * F_);
  }
  d_probe_weights_.allocate(D_);

  dm_update_object_->setDeviceBuffers(d_obj_.get(),
                                      d_obj_weights_.get(),
                                      d_probe_.get(),
                                      d_exit_wave_.get(),
                                      d_addr_info_.get(),
                                      d_cfact_object_.get());
  dm_update_object_->allocate();

  if (doUpdateProbe_)
  {
    dm_update_probe_->setDeviceBuffers(d_obj_.get(),
                                       d_probe_weights_.get(),
                                       d_probe_.get(),
                                       d_exit_wave_.get(),
                                       d_addr_info_.get(),
                                       d_cfact_probe_.get(),
                                       d_probe_support_.get());
    dm_update_probe_->allocate();
  }
  if (doCentering_)
  {
    center_probe_->setDeviceBuffers(d_probe_.get(), nullptr);
    center_probe_->allocate();
  }
}

void DifferenceMapOverlapConstraint::transfer_in(
    const int* addr_info,
    const complex<float>* cfact_object,
    const complex<float>* cfact_probe,
    const complex<float>* exit_wave,
    const complex<float>* obj,
    const float* obj_weigths,
    const complex<float>* probe,
    const complex<float>* probe_support,
    const float* probe_weights)
{
  ScopedTimer t(this, "transfer in");
  gpu_memcpy_h2d(d_addr_info_.get(), addr_info, A_ * 15);
  gpu_memcpy_h2d(d_cfact_object_.get(), cfact_object, G_ * H_ * I_);
  gpu_memcpy_h2d(d_cfact_probe_.get(), cfact_probe, D_ * E_ * F_);
  gpu_memcpy_h2d(d_exit_wave_.get(), exit_wave, A_ * B_ * C_);
  gpu_memcpy_h2d(d_obj_.get(), obj, G_ * H_ * I_);
  gpu_memcpy_h2d(d_obj_weights_.get(), obj_weigths, G_);
  gpu_memcpy_h2d(d_probe_.get(), probe, D_ * E_ * F_);
  if (withProbeSupport_)
  {
    gpu_memcpy_h2d(d_probe_support_.get(), probe_support, D_ * E_ * F_);
  }
  gpu_memcpy_h2d(d_probe_weights_.get(), probe_weights, D_);
}

void DifferenceMapOverlapConstraint::run(int max_iterations,
                                         float clip_min,
                                         float clip_max,
                                         float probe_center_tol,
                                         float overlap_converge_factor,
                                         bool do_update_probe)
{
  ScopedTimer t(this, "run");

  bool doUpdateProbeCombined = doUpdateProbe_ && do_update_probe;

  for (int inner = 0; inner < max_iterations; ++inner)
  {
    if (doUpdateObjectFirst_ || inner > 0)
    {
      dm_update_object_->run(clip_min, clip_max);
    }

    // exit if probe should not be updated yet
    if (!doUpdateProbeCombined)
    {
      break;
    }

    dm_update_probe_->run();
    float change = 0.0f;
    dm_update_probe_->transfer_out(nullptr, &change);

    // recenter the probe
    if (doCentering_)
    {
      center_probe_->run(probe_center_tol);
    }

    // stop iteration if probe change is small
    if (change < overlap_converge_factor)
      break;
  }

  timing_sync();
}

void DifferenceMapOverlapConstraint::transfer_out(complex<float>* probe,
                                                  complex<float>* obj)
{
  ScopedTimer t(this, "transfer out");
  gpu_memcpy_d2h(probe, d_probe_.get(), D_ * E_ * F_);
  gpu_memcpy_d2h(obj, d_obj_.get(), G_ * H_ * I_);
}

/********* interface functions *************/

extern "C" void difference_map_overlap_constraint_c(
    const int* addr_info,          // A x 5 x 3
    const float* f_cfact_object,   // G x H x I
    const float* f_cfact_probe,    // D x E x F
    const float* f_exit_wave,      // A x B x C
    float* f_obj,                  // G x H x I
    const float* object_weights,   // G
    float* f_probe,                // D x E x F
    const float* f_probe_support,  // D x E x F, can be null
    const float* probe_weights,    // D
    float obj_smooth_std,
    float clip_min,
    float clip_max,
    float probe_center_tol,
    float overlap_converge_factor,
    int max_iterations,
    int doUpdateObjectFirst,
    int doUpdateProbe,
    int doSmoothing,
    int doClipping,
    int doCentering,
    int A,
    int B,
    int C,
    int D,
    int E,
    int F,
    int G,
    int H,
    int I)
{
  auto obj = reinterpret_cast<complex<float>*>(f_obj);
  auto probe = reinterpret_cast<complex<float>*>(f_probe);
  auto probe_support = reinterpret_cast<const complex<float>*>(f_probe_support);
  auto cfact_object = reinterpret_cast<const complex<float>*>(f_cfact_object);
  auto cfact_probe = reinterpret_cast<const complex<float>*>(f_cfact_probe);
  auto exit_wave = reinterpret_cast<const complex<float>*>(f_exit_wave);

  auto dmoc = gpuManager.get_cuda_function<DifferenceMapOverlapConstraint>(
      "dm_overlap_constraint",
      A,
      B,
      C,
      D,
      E,
      F,
      G,
      H,
      I,
      obj_smooth_std,
      doUpdateObjectFirst != 0,
      doUpdateProbe != 0,
      doSmoothing != 0,
      doClipping != 0,
      probe_support != 0,
      doCentering != 0);
  dmoc->allocate();
  dmoc->transfer_in(addr_info,
                    cfact_object,
                    cfact_probe,
                    exit_wave,
                    obj,
                    object_weights,
                    probe,
                    probe_support,
                    probe_weights);
  dmoc->run(max_iterations,
            clip_min,
            clip_max,
            probe_center_tol,
            overlap_converge_factor);
  dmoc->transfer_out(probe, obj);
}