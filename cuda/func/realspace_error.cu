#include "realspace_error.h"

#include "utils/ScopedTimer.h"
/*************** kernels **********************/

__global__ void realspace_error_kernel(const complex<float> *difference,
                                       float *out,
                                       int m,
                                       int n)
{
  int batch = blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int txy = tx * blockDim.y + ty;

  extern __shared__ float sum[];
  sum[txy] = 0.0;

  for (int i = tx; i < m; i += blockDim.x)
  {
    for (int j = ty; j < n; j += blockDim.y)
    {
      auto idx = batch * m * n + i * n + j;
      auto v = difference[idx];
      auto abs2 = v.real() * v.real() + v.imag() * v.imag();
      sum[txy] += abs2;
    }
  }

  __syncthreads();
  int nt = blockDim.x * blockDim.y;
  int c = nt;
  while (c > 1)
  {
    int half = c / 2;
    if (txy < half)
    {
      sum[txy] += sum[c - txy - 1];
    }
    __syncthreads();
    c = c - half;
  }

  if (txy == 0)
  {
    out[batch] = sum[0] / float(m * n);
  }
}

/*************** class implementation ***************/

RealspaceError::RealspaceError(int i, int m, int n)
    : CudaFunction("realspace_error"), i_(i), m_(m), n_(n)
{
}

void RealspaceError::setDeviceBuffers(complex<float> *d_difference,
                                      float *d_out)
{
  d_difference_ = d_difference;
  d_out_ = d_out;
}

void RealspaceError::allocate()
{
  ScopedTimer t(this, "allocate");
  d_difference_.allocate(i_ * m_ * n_);
  d_out_.allocate(i_);
}

void RealspaceError::transfer_in(const complex<float> *difference)
{
  ScopedTimer t(this, "transfer in");
  gpu_memcpy_h2d(d_difference_.get(), difference, i_ * n_ * m_);
}

void RealspaceError::run()
{
  ScopedTimer t(this, "run");

  // always use a 32x32 block of threads
  dim3 threadsPerBlock = {32u, 32u, 1u};
  dim3 blocks = {unsigned(i_), 1u, 1u};
  realspace_error_kernel<<<blocks, threadsPerBlock, 32 * 32 * sizeof(float)>>>(
      d_difference_.get(), d_out_.get(), m_, n_);
  checkLaunchErrors();

  timing_sync();
}

void RealspaceError::transfer_out(float *out)
{
  ScopedTimer t(this, "transfer out");
  gpu_memcpy_d2h(out, d_out_.get(), i_);
}

/************** interface function ***************/

extern "C" void realspace_error_c(
    const float *difference, float *out, int i, int m, int n)
{
  RealspaceError rse(i, m, n);
  rse.allocate();
  rse.transfer_in(reinterpret_cast<const complex<float> *>(difference));
  rse.run();
  rse.transfer_out(out);
}