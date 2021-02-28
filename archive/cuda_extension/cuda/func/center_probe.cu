#include "center_probe.h"

#include "utils/GpuManager.h"
#include "utils/ScopedTimer.h"

// 1 block per 2nd/3rd dim
template <int BlockX, int BlockY>
__global__ void sum_abs2(const complex<float>* in,
                         float* out,
                         int dim0,
                         int dim12)
{
  int ty = threadIdx.y + blockIdx.y * BlockY;
  int tx = threadIdx.x;

  auto val = 0.0f;
  if (ty < dim12)
  {
    // specific block iterates over the 1st dim,
    // and has fixed x/y
    in += ty;
    for (int i = tx; i < dim0; i += BlockX)
    {
      auto cval = in[i * dim12];
      auto abs2 = cval.real() * cval.real() + cval.imag() * cval.imag();
      val += abs2;
    }
  }

  __shared__ float blocksums[BlockX][BlockY];
  blocksums[tx][threadIdx.y] = val;

  __syncthreads();
  int nt = blockDim.x;
  int c = nt;
  while (c > 1)
  {
    int half = c / 2;
    if (tx < half)
    {
      blocksums[tx][threadIdx.y] += blocksums[c - tx - 1][threadIdx.y];
    }
    __syncthreads();
    c = c - half;
  }

  if (ty >= dim12)
    return;

  if (tx == 0)
    out[ty] = blocksums[0][threadIdx.y];
}

/****************** class implementation ***********/

CenterProbe::CenterProbe() : CudaFunction("center_probe") {}

void CenterProbe::setParameters(int i, int m, int n)
{
  i_ = i;
  m_ = m;
  n_ = n;

  mass_center_ = gpuManager.get_cuda_function<MassCenter>(
      "center_probe.mass_center", m_, n_, 1);
  interp_shift_ = gpuManager.get_cuda_function<InterpolatedShift>(
      "center_probe.interpolated_shift", i_, m_, n_);
}

void CenterProbe::setDeviceBuffers(complex<float>* d_probe,
                                   complex<float>* d_out)
{
  d_probe_ = d_probe;
  d_out_ = d_out;
}

void CenterProbe::allocate()
{
  ScopedTimer t(this, "allocate");
  d_probe_.allocate(i_ * m_ * n_);
  d_out_.allocate(i_ * m_ * n_);
  d_buffer_.allocate(m_ * n_);
  mass_center_->setDeviceBuffers(d_buffer_.get(), nullptr);
  mass_center_->allocate();
  interp_shift_->setDeviceBuffers(d_probe_.get(), d_out_.get());
  interp_shift_->allocate();
}

void CenterProbe::transfer_in(const complex<float>* probe)
{
  ScopedTimer t(this, "transfer in");
  gpu_memcpy_h2d(d_probe_.get(), probe, i_ * m_ * n_);
}

void CenterProbe::transfer_out(complex<float>* probe)
{
  ScopedTimer t(this, "transfer out");
  gpu_memcpy_d2h(probe, d_probe_.get(), i_ * m_ * n_);
}

void CenterProbe::run(float center_tolerance)
{
  ScopedTimer t(this, "run");

  dim3 threads = {32u, 32u, 1u};
  dim3 blocks = {1u, unsigned(m_ * n_ + 32 - 1) / 32u, 1u};
  sum_abs2<32, 32>
      <<<blocks, threads>>>(d_probe_.get(), d_buffer_.get(), i_, m_ * n_);
  checkLaunchErrors();

  // now mass_center across the buffer
  mass_center_->run();

  // TODO: see if this can be done on the GPU directly,
  // avoiding a sync point in chain of async kernels
  // Note: means putting interp_shift offset as a device buffer,
  // not as parameter to the run function itself
  float c1[2];
  mass_center_->transfer_out(c1);
  float c2[] = {float(m_ / 2), float(n_ / 2)};
  auto err_1 = c1[0] - c2[0];
  auto err_2 = c1[1] - c2[1];
  auto err = std::sqrt(err_1 * err_1 + err_2 * err_2);

  if (err < center_tolerance)
  {
    return;
  }

  float offset[] = {c2[0] - c1[0], c2[1] - c1[1]};

  // now interpolated_shift on the data, and back into probe array
  interp_shift_->run(offset[0], offset[1], true);
  gpu_memcpy_d2d(d_probe_.get(), interp_shift_->getOutput(), i_ * m_ * n_);

  timing_sync();
}

/************ interface functions *************/

extern "C" void center_probe_c(
    float* f_probe, float center_tolerance, int i, int m, int n)
{
  auto probe = reinterpret_cast<complex<float>*>(f_probe);
  auto cp = gpuManager.get_cuda_function<CenterProbe>("center_probe", i, m, n);
  cp->allocate();
  cp->transfer_in(probe);
  cp->run(center_tolerance);
  cp->transfer_out(probe);
}