#include "difference_map_iterator.h"

#include "utils/Errors.h"
#include "utils/GpuManager.h"
#include "utils/ScopedTimer.h"

#include "addr_info_helpers.h"

/************** class implementation *********/

DifferenceMapIterator::DifferenceMapIterator()
    : CudaFunction("difference_map_iterator")
{
}

void DifferenceMapIterator::setParameters(int A,
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
                                          bool doProbeSupport)
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
  N_ = N;
  num_iterations_ = num_iterations;
  doPbound_ = doPbound;
  doProbeSupport_ = doProbeSupport;

  // TODO: shall we make the realspace error a parameter too?
  dm_fourier_constraint_ =
      gpuManager.get_cuda_function<DifferenceMapFourierConstraint>(
          "dm_iterator.dm_fourier_constraint",
          A,
          N,
          E,  // E == B
          F,  // F == C
          H,
          I,
          G,
          D,
          do_LL_error,
          doRealspaceError);
  dm_overlap_update_ =
      gpuManager.get_cuda_function<DifferenceMapOverlapConstraint>(
          "dm_iterator.dm_overlap_update",
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
          doUpdateObjectFirst,
          true,  // in general, allocate mem, etc for probe update
          doSmoothing,
          doClipping,
          doProbeSupport,
          doCentering);
}

int DifferenceMapIterator::calculateAddrIndices(const int* out1_addr)
{
  // calculate the indexing map
  outidx_.clear();
  startidx_.clear();
  indices_.clear();
  flatten_out_addr(out1_addr, A_, 15, outidx_, startidx_, indices_);
  outidx_size_ = outidx_.size();
  return outidx_size_;
}

void DifferenceMapIterator::calculateUniqueDaIndices(const int* da_addr)
{
  dm_fourier_constraint_->calculateUniqueDaIndices(da_addr);
}

void DifferenceMapIterator::setDeviceBuffers(float* d_diffraction,
                                             complex<float>* d_obj,
                                             float* d_object_weights,
                                             complex<float>* d_cfact_object,
                                             unsigned char* d_mask,
                                             complex<float>* d_probe,
                                             complex<float>* d_cfact_probe,
                                             complex<float>* d_probe_support,
                                             float* d_probe_weights,
                                             complex<float>* d_exit_wave,
                                             int* d_addr_info,
                                             complex<float>* d_pre_fft,
                                             complex<float>* d_post_fft,
                                             float* d_errors)
{
  d_diffraction_ = d_diffraction;
  d_obj_ = d_obj;
  d_object_weights_ = d_object_weights;
  d_cfact_object_ = d_cfact_object;
  d_mask_ = d_mask;
  d_probe_ = d_probe;
  d_cfact_probe_ = d_cfact_probe;
  d_probe_support_ = d_probe_support;
  d_probe_weights_ = d_probe_weights;
  d_exit_wave_ = d_exit_wave;
  d_addr_info_ = d_addr_info;
  d_pre_fft_ = d_pre_fft;
  d_post_fft_ = d_post_fft;
  d_errors_ = d_errors;
}

void DifferenceMapIterator::allocate()
{
  ScopedTimer t(this, "allocate");

  d_diffraction_.allocate(N_ * B_ * C_);
  d_obj_.allocate(G_ * H_ * I_);
  d_object_weights_.allocate(G_);
  d_cfact_object_.allocate(G_ * H_ * I_);
  d_mask_.allocate(N_ * B_ * C_);
  d_probe_.allocate(D_ * E_ * F_);
  if (doProbeSupport_)
  {
    d_probe_support_.allocate(D_ * E_ * F_);
  }
  d_probe_weights_.allocate(D_);
  d_exit_wave_.allocate(A_ * B_ * C_);
  d_addr_info_.allocate(A_ * 15);
  d_pre_fft_.allocate(C_ * B_);
  d_post_fft_.allocate(C_ * B_);
  d_errors_.allocate(num_iterations_ * 3 * N_);

  if (!outidx_.empty())
  {
    d_outidx_.allocate(outidx_.size());
    d_startidx_.allocate(startidx_.size());
    d_indices_.allocate(indices_.size());
  }

  dm_fourier_constraint_->setDeviceBuffers(d_mask_.get(),
                                           d_diffraction_.get(),
                                           d_obj_.get(),
                                           d_probe_.get(),
                                           d_exit_wave_.get(),
                                           d_addr_info_.get(),
                                           d_pre_fft_.get(),
                                           d_post_fft_.get(),
                                           d_errors_.get(),
                                           d_outidx_.get(),
                                           d_startidx_.get(),
                                           d_indices_.get(),
                                           outidx_size_);
  dm_fourier_constraint_->allocate();

  dm_overlap_update_->setDeviceBuffers(d_addr_info_.get(),
                                       d_cfact_object_.get(),
                                       d_cfact_probe_.get(),
                                       d_exit_wave_.get(),
                                       d_obj_.get(),
                                       d_object_weights_.get(),
                                       d_probe_.get(),
                                       d_probe_support_.get(),
                                       d_probe_weights_.get());
  dm_overlap_update_->allocate();
}

void DifferenceMapIterator::transfer_in(const float* diffraction,
                                        const complex<float>* obj,
                                        const float* object_weights,
                                        const complex<float>* cfact_object,
                                        const unsigned char* mask,
                                        const complex<float>* probe,
                                        const complex<float>* cfact_probe,
                                        const complex<float>* probe_support,
                                        const float* probe_weights,
                                        const complex<float>* exit_wave,
                                        const int* addr_info,
                                        const complex<float>* pre_fft,
                                        const complex<float>* post_fft)
{
  ScopedTimer t(this, "transfer in");
  gpu_memcpy_h2d(d_diffraction_.get(), diffraction, N_ * B_ * C_);
  gpu_memcpy_h2d(d_obj_.get(), obj, G_ * H_ * I_);
  gpu_memcpy_h2d(d_object_weights_.get(), object_weights, G_);
  gpu_memcpy_h2d(d_cfact_object_.get(), cfact_object, G_ * H_ * I_);
  gpu_memcpy_h2d(d_mask_.get(), mask, N_ * B_ * C_);
  gpu_memcpy_h2d(d_probe_.get(), probe, D_ * E_ * F_);
  gpu_memcpy_h2d(d_cfact_probe_.get(), cfact_probe, D_ * E_ * F_);
  if (doProbeSupport_)
  {
    gpu_memcpy_h2d(d_probe_support_.get(), probe_support, D_ * E_ * F_);
  }
  gpu_memcpy_h2d(d_probe_weights_.get(), probe_weights, D_);
  gpu_memcpy_h2d(d_exit_wave_.get(), exit_wave, A_ * B_ * C_);
  gpu_memcpy_h2d(d_addr_info_.get(), addr_info, A_ * 15);
  gpu_memcpy_h2d(d_pre_fft_.get(), pre_fft, B_ * C_);
  gpu_memcpy_h2d(d_post_fft_.get(), post_fft, B_ * C_);

  if (!outidx_.empty())
  {
    gpu_memcpy_h2d(d_outidx_.get(), outidx_.data(), outidx_.size());
    gpu_memcpy_h2d(d_startidx_.get(), startidx_.data(), startidx_.size());
    gpu_memcpy_h2d(d_indices_.get(), indices_.data(), indices_.size());
  }

  calculateUniqueDaIndices(addr_info + 9);
}

void DifferenceMapIterator::run(int overlap_max_iterations,
                                float overlap_converge_factor,
                                float probe_center_tol,
                                int probe_update_start,
                                float pbound,
                                float alpha,
                                float clip_min,
                                float clip_max)
{
  ScopedTimer t(this, "run");

  for (int it = 0; it < num_iterations_; ++it)
  {
    if (((it + 1) % 10 == 0) && it > 0)
    {
      std::cout << "iteration: " << it + 1 << std::endl;
    }

    dm_fourier_constraint_->updateErrorOutput(d_errors_.get() + it * 3 * N_);
    dm_fourier_constraint_->run(pbound, alpha, doPbound_);

    auto do_update_probe = probe_update_start <= it;
    dm_overlap_update_->run(overlap_max_iterations,
                            clip_min,
                            clip_max,
                            probe_center_tol,
                            overlap_converge_factor,
                            do_update_probe);
  }

  timing_sync();
}

void DifferenceMapIterator::transfer_out(float* errors,
                                         complex<float>* obj,
                                         complex<float>* probe,
                                         complex<float>* exit_wave)
{
  ScopedTimer t(this, "transfer out");
  gpu_memcpy_d2h(errors, d_errors_.get(), num_iterations_ * 3 * N_);
  gpu_memcpy_d2h(obj, d_obj_.get(), G_ * H_ * I_);
  gpu_memcpy_d2h(probe, d_probe_.get(), D_ * E_ * F_);
  gpu_memcpy_d2h(exit_wave, d_exit_wave_.get(), A_ * B_ * C_);
}

/************** interface *************/

extern "C" void difference_map_iterator_c(
    // note: E = B, F = C
    const float* diffraction,    // N x B x C
    float* f_obj,                  // G x H x I
    const float* object_weights,   // G
    const float* f_cfact_object,   // G x H x I
    const unsigned char* mask,     // N x B x C
    float* f_probe,                // D x E x F
    const float* f_cfact_probe,    // D x E x F
    const float* f_probe_support,  // D x E x F
    const float* probe_weights,    // D
    float* f_exit_wave,            // A x B x C
    const int* addr_info,          // A x 5 x 3
    const float* f_pre_fft,        // B x C
    const float* f_post_fft,       // B x C
    float* errors,                 // num_iterations x 3 x N
    float pbound,
    int overlap_max_iterations,
    int doUpdateObjectFirst,
    float obj_smooth_std,
    float overlap_converge_factor,
    float probe_center_tol,
    int probe_update_start,
    float alpha,
    float clip_min,
    float clip_max,
    int do_LL_error,
    int do_realspace_error,
    int num_iterations,
    int A,
    int B,
    int C,
    int D,
    int E,
    int F,
    int G,
    int H,
    int I,
    int N,
    int doSmoothing,
    int doClipping,
    int doCentering,
    int doPbound)
{
  auto obj = reinterpret_cast<complex<float>*>(f_obj);
  auto cfact_object = reinterpret_cast<const complex<float>*>(f_cfact_object);
  auto probe = reinterpret_cast<complex<float>*>(f_probe);
  auto cfact_probe = reinterpret_cast<const complex<float>*>(f_cfact_probe);
  auto probe_support = reinterpret_cast<const complex<float>*>(f_probe_support);
  auto exit_wave = reinterpret_cast<complex<float>*>(f_exit_wave);
  auto pre_fft = reinterpret_cast<const complex<float>*>(f_pre_fft);
  auto post_fft = reinterpret_cast<const complex<float>*>(f_post_fft);

  if (E != B || F != C)
  {
    throw std::runtime_error("2nd/3rd dimensions of probe and mask are not consistent");
  }

  auto dmi = gpuManager.get_cuda_function<DifferenceMapIterator>(
      "dm_iterator",
      A,
      B,
      C,
      D,
      E,
      F,
      G,
      H,
      I,
      N,
      num_iterations,
      obj_smooth_std,
      doSmoothing != 0,
      doClipping != 0,
      doCentering != 0,
      doPbound != 0,
      do_LL_error != 0,
      do_realspace_error != 0,
      doUpdateObjectFirst != 0,
      probe_support != nullptr);
  dmi->calculateAddrIndices(addr_info + 9);
  dmi->allocate();
  dmi->transfer_in(diffraction,
                   obj,
                   object_weights,
                   cfact_object,
                   mask,
                   probe,
                   cfact_probe,
                   probe_support,
                   probe_weights,
                   exit_wave,
                   addr_info,
                   pre_fft,
                   post_fft);
  dmi->run(overlap_max_iterations,
           overlap_converge_factor,
           probe_center_tol,
           probe_update_start,
           pbound,
           alpha,
           clip_min,
           clip_max);
  dmi->transfer_out(errors, obj, probe, exit_wave);
}