#include "difference_map_fourier_constraint.h"

#include "addr_info_helpers.h"
#include "utils/GpuManager.h"
#include "utils/ScopedTimer.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

/*************** kernels ********************/

template <class T>
__global__ void add_inplace_kernel(T *inout, const T *a, int n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n)
    return;
  inout[tid] += a[tid];
}

template <class T1, class T2>
__global__ void div_inplace_kernel(T1 *inout, T2 divisor, int n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n)
    return;
  inout[tid] /= divisor;
}

template <class T>
__global__ void sqrt_abs_kernel(const T *in, T *out, int n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n)
    return;
  using std::abs;
  using std::sqrt;
  out[tid] = sqrt(abs(in[tid]));
}

template <class T>
__global__ void sqrt_kernel(const T *in, T *out, int n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n)
    return;
  using std::sqrt;
  out[tid] = sqrt(in[tid]);
}

template <class T>
__global__ void conjugate_kernel(const complex<T> *in, complex<T> *out, int n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n)
    return;
  out[tid] = complex<T>(in[tid].real(), -in[tid].imag());
}

/*************** class implementation ***********/

DifferenceMapFourierConstraint::DifferenceMapFourierConstraint()
    : CudaFunction("difference_map_fourier_constraint")
{
}

void DifferenceMapFourierConstraint::setParameters(int M,
                                                   int N,
                                                   int A,
                                                   int B,
                                                   int C,
                                                   int D,
                                                   int ob_modes,
                                                   int pr_modes,
                                                   bool do_LL_error,
                                                   bool do_realspace_error)
{
  M_ = M;
  N_ = N;
  A_ = A;
  B_ = B;
  C_ = C;
  D_ = D;
  pr_modes_ = pr_modes;
  ob_modes_ = ob_modes;
  do_LL_error_ = do_LL_error;
  do_realspace_error_ = do_realspace_error;

  scan_and_multiply_ = gpuManager.get_cuda_function<ScanAndMultiply>(
      "dm_fourier_contraint.scan_and_multiply",
      M_,
      A_,
      B_,
      pr_modes_,
      A_,
      B_,
      ob_modes_,
      C_,
      D_,
      M_);

  log_likelihood_ = gpuManager.get_cuda_function<LogLikelihood>(
      "dm_fourier_contraint.log_likelihood", M_, A_, B_, M_, N_);

  difference_map_realspace_constraint_ =
      gpuManager.get_cuda_function<DifferenceMapRealspaceConstraint>(
          "dm_fourier_constraint.dm_realspace_constraint", M_, A_, B_);

  farfield_propagator_fwd_ = gpuManager.get_cuda_function<FarfieldPropagator>(
      "dm_fourier_constraint.farfield_propagator_fwd", M_, A_, B_);

  abs2_ = gpuManager.get_cuda_function<Abs2<complex<float>, float>>(
      "dm_fourier_contraint.abs2", M_ * A_ * B_);

  sum_to_buffer_ = gpuManager.get_cuda_function<SumToBuffer<float>>(
      "dm_fourier_constraint.sum2buffer", M_, A_, B_, N_, A_, B_, M_, M_);

  far_field_error_ = gpuManager.get_cuda_function<FarFieldError>(
      "dm_fourier_constraint.farfield_error", N_, A_, B_);

  renormalise_fourier_magnitudes_ =
      gpuManager.get_cuda_function<RenormaliseFourierMagnitudes>(
          "dm_fourier_constraint.renormalise_fourier_magnitudes",
          M_,
          N_,
          A_,
          B_);

  farfield_propagator_rev_ = gpuManager.get_cuda_function<FarfieldPropagator>(
      "dm_fourier_constraint.farfield_propagator_rev", M_, A_, B_);

  get_difference_ = gpuManager.get_cuda_function<GetDifference>(
      "dm_fourier_constraint.get_difference", M_, A_, B_);

  realspace_error_ = gpuManager.get_cuda_function<RealspaceError>(
      "dm_fourier_constraint.realspace_error", M_, A_, B_, M_, N_);

  sum_to_buffer_->setAddrStride(15);
  realspace_error_->setAddrStride(15);
}

void DifferenceMapFourierConstraint::setDeviceBuffers(
    unsigned char *d_mask,
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
    int outidx_size)
{
  d_mask_ = d_mask;
  d_Idata_ = d_Idata;
  d_obj_ = d_obj;
  d_probe_ = d_probe;
  d_exit_wave_ = d_exit_wave;
  d_addr_info_ = d_addr_info;
  d_prefilter_ = d_prefilter;
  d_postfilter_ = d_postfilter;
  d_errors_ = d_errors;
  d_outidx_ = d_outidx;
  d_startidx_ = d_startidx;
  d_indices_ = d_indices;
  outidx_size_ = outidx_size;
}

int DifferenceMapFourierConstraint::calculateAddrIndices(const int *out1_addr)
{
  // calculate the indexing map
  outidx_.clear();
  startidx_.clear();
  indices_.clear();
  flatten_out_addr(out1_addr, M_, 15, outidx_, startidx_, indices_);
  outidx_size_ = outidx_.size();
  return outidx_size_;
}

void DifferenceMapFourierConstraint::calculateUniqueDaIndices(
    const int *da_addr)
{
  if (do_LL_error_)
  {
    log_likelihood_->calculateUniqueDaIndices(da_addr);
  }
}

void DifferenceMapFourierConstraint::updateErrorOutput(float *d_errors)
{
  d_errors_ = d_errors;
  checkCudaErrors(
      cudaMemset(d_errors_.get(), 0, N_ * 3 * sizeof(*d_errors_.get())));
  if (do_LL_error_)
  {
    log_likelihood_->updateErrorOutput(d_errors_.get() + N_);
  }
  if (do_realspace_error_)
  {
    realspace_error_->updateErrorOutput(d_errors_.get() + 2 * N_);
  }
  far_field_error_->updateErrorOutput(d_errors_.get());
  get_difference_->updateErrorInput(d_errors_.get());
  renormalise_fourier_magnitudes_->updateErrorInput(d_errors_.get());
}

void DifferenceMapFourierConstraint::allocate()
{
  ScopedTimer t(this, "allocate (joint)");

  d_mask_.allocate(M_ * A_ * B_);
  d_Idata_.allocate(N_ * A_ * B_);
  d_obj_.allocate(ob_modes_ * C_ * D_);
  d_probe_.allocate(pr_modes_ * A_ * B_);
  d_exit_wave_.allocate(M_ * A_ * B_);
  d_addr_info_.allocate(M_ * 3 * 5);
  d_prefilter_.allocate(A_ * B_);
  d_postfilter_.allocate(A_ * B_);
  d_prefilter_conj_.allocate(A_ * B_);
  d_postfilter_conj_.allocate(A_ * B_);
  d_errors_.allocate(N_ * 3);
  checkCudaErrors(
      cudaMemset(d_errors_.get(), 0, N_ * 3 * sizeof(*d_errors_.get())));
  d_fmag_.allocate(M_ * A_ * B_);
  if (!outidx_.empty())
  {
    d_outidx_.allocate(outidx_.size());
    d_startidx_.allocate(startidx_.size());
    d_indices_.allocate(indices_.size());
  }

  // probe, obj, addr_info --> probe_obj
  scan_and_multiply_->setDeviceBuffers(
      d_probe_.get(),
      d_obj_.get(),
      d_addr_info_.get(),
      nullptr  // the output buffer is allocated
  );
  scan_and_multiply_->allocate();
  auto d_probe_obj = scan_and_multiply_->getOutput();

  if (do_LL_error_)
  {
    // probe_object, mask, Idata, prefilter, postfilter, addr --> err_phot
    log_likelihood_->setDeviceBuffers(d_probe_obj,
                                      d_mask_.get(),
                                      d_Idata_.get(),
                                      d_prefilter_.get(),
                                      d_postfilter_.get(),
                                      d_addr_info_.get(),
                                      d_errors_.get() + N_,
                                      d_outidx_.get(),
                                      d_startidx_.get(),
                                      d_indices_.get(),
                                      outidx_size_);
    log_likelihood_->allocate();
  }

  // probe_obj, exit_wave -> constrained
  difference_map_realspace_constraint_->setDeviceBuffers(
      d_probe_obj,
      d_exit_wave_.get(),
      nullptr  // the output is allocated in there
  );
  difference_map_realspace_constraint_->allocate();
  auto d_constrained = difference_map_realspace_constraint_->getOutput();

  farfield_propagator_fwd_->setDeviceBuffers(
      d_constrained,
      d_constrained,  // in-place, ok here
      d_prefilter_.get(),
      d_postfilter_.get());
  farfield_propagator_fwd_->allocate();
  // output is in d_constrained
  auto d_f = d_constrained;

  abs2_->setDeviceBuffers(d_f, nullptr);
  abs2_->allocate();

  // abs2f, idata shape, ea, da => af2
  // giving strided access to addr_info to avoid another copy
  // (constructor sets the stride to 15 instead of 3, we
  // just offset it here to get to the ea and da parts)
  sum_to_buffer_->setDeviceBuffers(abs2_->getOutput(),
                                   nullptr,  // output is allocated in here
                                   d_addr_info_.get() + 6,
                                   d_addr_info_.get() + 9,
                                   d_outidx_.get(),
                                   d_startidx_.get(),
                                   d_indices_.get(),
                                   outidx_size_);
  sum_to_buffer_->allocate();
  auto d_af2 = sum_to_buffer_->getOutput();

  // we'll run sqrt(af2) in-place
  auto d_af = d_af2;

  // d_fmag = sqrt(abs(Idata)) will be run in a kernel

  // af, fmag, mask => err_fmag
  far_field_error_->setDeviceBuffers(
      d_af, d_fmag_.get(), d_mask_.get(), d_errors_.get());
  far_field_error_->allocate();
  auto d_err_fmag = d_errors_.get();

  // f, af, fmag, mask, err_fmag, addr_info, pbound => vectorised_rfm
  renormalise_fourier_magnitudes_->setDeviceBuffers(
      d_f,
      d_af,
      d_fmag_.get(),
      d_mask_.get(),
      d_err_fmag,
      d_addr_info_.get(),
      nullptr  // gets allocated inside
  );
  renormalise_fourier_magnitudes_->allocate();
  auto d_vectorised_rfm = renormalise_fourier_magnitudes_->getOutput();

  // vectorised_rfm, postfilter.conj, prefilter.conj, 'reverse' ->
  // backpropagated_solution (flipped / conjugated post/pre filter)
  farfield_propagator_rev_->setDeviceBuffers(d_vectorised_rfm,
                                             d_vectorised_rfm,
                                             d_postfilter_conj_.get(),
                                             d_prefilter_conj_.get());
  farfield_propagator_rev_->allocate();
  auto d_backpropagated_solution = d_vectorised_rfm;

  // addr_info, alpha, backpropagated_solution, err_fmag, pbound, probe_object
  // -> df
  get_difference_->setDeviceBuffers(d_addr_info_.get(),
                                    d_backpropagated_solution,
                                    d_err_fmag,
                                    d_exit_wave_.get(),
                                    d_probe_obj,
                                    nullptr);
  get_difference_->allocate();
  auto d_df = get_difference_->getOutput();

  // we'll add df to exit_wave in-place

  if (do_realspace_error_)
  {
    realspace_error_->setDeviceBuffers(d_df,
                                       d_addr_info_.get() + 2 * 3,
                                       d_addr_info_.get() + 3 * 3,
                                       d_errors_.get() + 2 * N_);
    realspace_error_->allocate();
  }

  // we'll div by pbound in-place for d_err_fmag
}

void DifferenceMapFourierConstraint::transfer_in(
    const unsigned char *mask,
    const float *Idata,
    const complex<float> *obj,
    const complex<float> *probe,
    const complex<float> *exit_wave,
    const int *addr_info,
    const complex<float> *prefilter,
    const complex<float> *postfilter)
{
  ScopedTimer t(this, "transfer in");

  gpu_memcpy_h2d(d_mask_.get(), mask, N_ * A_ * B_);
  gpu_memcpy_h2d(d_Idata_.get(), Idata, N_ * A_ * B_);
  gpu_memcpy_h2d(d_obj_.get(), obj, ob_modes_ * C_ * D_);
  gpu_memcpy_h2d(d_probe_.get(), probe, pr_modes_ * A_ * B_);
  gpu_memcpy_h2d(d_exit_wave_.get(), exit_wave, M_ * A_ * B_);
  gpu_memcpy_h2d(d_addr_info_.get(), addr_info, M_ * 3 * 5);
  gpu_memcpy_h2d(d_prefilter_.get(), prefilter, A_ * B_);
  gpu_memcpy_h2d(d_postfilter_.get(), postfilter, A_ * B_);
  // conjugate will be run first-time in the run function

  if (!outidx_.empty())
  {
    gpu_memcpy_h2d(d_outidx_.get(), outidx_.data(), outidx_.size());
    gpu_memcpy_h2d(d_startidx_.get(), startidx_.data(), startidx_.size());
    gpu_memcpy_h2d(d_indices_.get(), indices_.data(), indices_.size());
  }

  calculateUniqueDaIndices(addr_info + 9);
}

void DifferenceMapFourierConstraint::run(float pbound,
                                         float alpha,
                                         bool doPbound)
{
  ScopedTimer t(this, "run");

  // conjugate the pre- and post-filters
  // TODO: do this only once if called multiple times
  int total = A_ * B_;
  int threadsPerBlock = 256;
  int blocks = (total + threadsPerBlock - 1) / threadsPerBlock;
  if (d_prefilter_.get() != nullptr)
  {
    conjugate_kernel<<<blocks, threadsPerBlock>>>(
        d_prefilter_.get(), d_prefilter_conj_.get(), total);
    checkLaunchErrors();
  }
  if (d_postfilter_.get() != nullptr)
  {
    conjugate_kernel<<<blocks, threadsPerBlock>>>(
        d_postfilter_.get(), d_postfilter_conj_.get(), total);
    checkLaunchErrors();
  }

  scan_and_multiply_->run();

  if (do_LL_error_)
  {
    log_likelihood_->run();
  }
  difference_map_realspace_constraint_->run(alpha);
  farfield_propagator_fwd_->run(
      d_prefilter_.get() != nullptr, d_postfilter_.get() != nullptr, true);
  abs2_->run();
  sum_to_buffer_->run();

  // sqrt(abs(Idata))
  total = N_ * A_ * B_;
  threadsPerBlock = 256;
  blocks = (total + threadsPerBlock - 1) / threadsPerBlock;
  sqrt_abs_kernel<<<blocks, threadsPerBlock>>>(
      d_Idata_.get(), d_fmag_.get(), total);
  checkLaunchErrors();

  // sqrt(af2), in-place
  sqrt_kernel<<<blocks, threadsPerBlock>>>(
      sum_to_buffer_->getOutput(), sum_to_buffer_->getOutput(), total);
  checkLaunchErrors();

  far_field_error_->run();
  renormalise_fourier_magnitudes_->run(pbound, doPbound);

  farfield_propagator_rev_->run(
      d_prefilter_.get() != nullptr, d_postfilter_.get() != nullptr, false);

  get_difference_->run(alpha, pbound, doPbound);

  auto df = get_difference_->getOutput();

  total = M_ * A_ * B_;
  threadsPerBlock = 256;
  blocks = (total + threadsPerBlock - 1) / threadsPerBlock;

  add_inplace_kernel<<<blocks, threadsPerBlock>>>(
      d_exit_wave_.get(), df, total);
  checkLaunchErrors();

  if (do_realspace_error_)
  {
    realspace_error_->run();
  }

  if (doPbound)
  {
    auto d_err_fmag = d_errors_.get();
    total = N_;
    threadsPerBlock = 256;
    blocks = (total + threadsPerBlock - 1) / threadsPerBlock;
    div_inplace_kernel<<<blocks, threadsPerBlock>>>(d_err_fmag, pbound, total);
    checkLaunchErrors();
  }

  timing_sync();
}
void DifferenceMapFourierConstraint::transfer_out(float *errors,
                                                  complex<float> *exit_wave)
{
  ScopedTimer t(this, "transfer out");
  gpu_memcpy_d2h(errors, d_errors_.get(), 3 * N_);
  gpu_memcpy_d2h(exit_wave, d_exit_wave_.get(), M_ * A_ * B_);
}

/**************** interface function ***********/

extern "C" void difference_map_fourier_constraint_c(
    const unsigned char *mask,  // N x A x B
    const float *Idata,         // N x A x B
    const float *f_obj,         // ob_modes x C x D
    const float *f_probe,       // pr_modes x A x B
    float *f_exit_wave,         // M x A x B
    const int *addr_info,       // M x 5 x 3
    const float *f_prefilter,   // A x B
    const float *f_postfilter,  // A x B
    float pbound,
    float alpha,
    int do_LL_error,
    int do_realspace_error,
    int doPbound,
    int M,
    int N,
    int A,
    int B,
    int C,
    int D,
    int ob_modes,
    int pr_modes,
    float *errors)
{
  auto obj = reinterpret_cast<const complex<float> *>(f_obj);
  auto probe = reinterpret_cast<const complex<float> *>(f_probe);
  auto exit_wave = reinterpret_cast<complex<float> *>(f_exit_wave);
  auto prefilter = reinterpret_cast<const complex<float> *>(f_prefilter);
  auto postfilter = reinterpret_cast<const complex<float> *>(f_postfilter);

  auto dmfc = gpuManager.get_cuda_function<DifferenceMapFourierConstraint>(
      "dm_fourier_constraint",
      M,
      N,
      A,
      B,
      C,
      D,
      ob_modes,
      pr_modes,
      do_LL_error != 0,
      do_realspace_error != 0);

  dmfc->calculateAddrIndices(addr_info + 9);
  dmfc->allocate();
  dmfc->transfer_in(
      mask, Idata, obj, probe, exit_wave, addr_info, prefilter, postfilter);
  dmfc->run(pbound, alpha, doPbound != 0);
  dmfc->transfer_out(errors, exit_wave);
}