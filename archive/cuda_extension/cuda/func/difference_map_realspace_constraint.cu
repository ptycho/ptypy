#include "difference_map_realspace_constraint.h"

#include "utils/Complex.h"
#include "utils/GpuManager.h"
#include "utils/ScopedTimer.h"

/************ Kernels ******************/

__global__ void difference_map_realspace_constraint_kernel(
    const complex<float> *obj_and_probe,
    const complex<float> *exit_wave,
    float alpha,
    complex<float> *out,
    size_t total)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;
  auto pno = obj_and_probe[idx];
  auto ex = exit_wave[idx];
  auto val = (1.0f + alpha) * pno - alpha * ex;
  out[idx] = val;
}

/***************** Class implementation ***********/

DifferenceMapRealspaceConstraint::DifferenceMapRealspaceConstraint()
    : CudaFunction("difference_map_realspace_constraint")
{
}

void DifferenceMapRealspaceConstraint::setParameters(int i, int m, int n)
{
  i_ = i;
  m_ = m;
  n_ = n;
}

void DifferenceMapRealspaceConstraint::setDeviceBuffers(
    complex<float> *d_obj_and_probe,
    complex<float> *d_exit_wave,
    complex<float> *d_out)
{
  d_obj_and_probe_ = d_obj_and_probe;
  d_exit_wave_ = d_exit_wave;
  d_out_ = d_out;
}

void DifferenceMapRealspaceConstraint::allocate()
{
  ScopedTimer t(this, "allocate");
  d_obj_and_probe_.allocate(i_ * m_ * n_);
  d_exit_wave_.allocate(i_ * m_ * n_);
  d_out_.allocate(i_ * m_ * n_);
}

complex<float> *DifferenceMapRealspaceConstraint::getOutput() const
{
  return d_out_.get();
}

void DifferenceMapRealspaceConstraint::transfer_in(
    const complex<float> *obj_and_probe, const complex<float> *exit_wave)
{
  ScopedTimer t(this, "transfer in");
  gpu_memcpy_h2d(d_obj_and_probe_.get(), obj_and_probe, i_ * m_ * n_);
  gpu_memcpy_h2d(d_exit_wave_.get(), exit_wave, i_ * m_ * n_);
}

void DifferenceMapRealspaceConstraint::transfer_out(complex<float> *out)
{
  ScopedTimer t(this, "transfer out");
  gpu_memcpy_d2h(out, d_out_.get(), i_ * m_ * n_);
}

void DifferenceMapRealspaceConstraint::run(float alpha)
{
  ScopedTimer t(this, "run");
  size_t total = size_t(m_) * size_t(n_) * size_t(i_);
  size_t block = 256;
  size_t blocks = (total + block - 1) / block;

  difference_map_realspace_constraint_kernel<<<blocks, block>>>(
      d_obj_and_probe_.get(), d_exit_wave_.get(), alpha, d_out_.get(), total);
  checkLaunchErrors();

  // sync device if timing is enabled
  timing_sync();
}

/**************** interface function ************/

extern "C" void difference_map_realspace_constraint_c(
    const float *fobj_and_probe,
    const float *f_exit_wave,
    float alpha,
    int i,
    int m,
    int n,
    float *fout)
{
  auto obj_and_probe = reinterpret_cast<const complex<float> *>(fobj_and_probe);
  auto exit_wave = reinterpret_cast<const complex<float> *>(f_exit_wave);
  auto out = reinterpret_cast<complex<float> *>(fout);

  auto dmc = gpuManager.get_cuda_function<DifferenceMapRealspaceConstraint>(
      "dm_realspace_constraint", i, m, n);
  dmc->allocate();
  dmc->transfer_in(obj_and_probe, exit_wave);
  dmc->run(alpha);
  dmc->transfer_out(out);
}