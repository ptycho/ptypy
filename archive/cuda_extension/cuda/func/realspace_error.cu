#include "realspace_error.h"

#include "utils/GpuManager.h"
#include "utils/ScopedTimer.h"

/*************** kernels **********************/

__global__ void realspace_error_kernel(const complex<float> *difference,
                                       const int *ea_first_column,
                                       const int *da_first_column,
                                       int addr_stride,
                                       float *out,
                                       int m,
                                       int n)
{
  int bid = blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int txy = tx * blockDim.y + ty;

  extern __shared__ float sum[];
  sum[txy] = 0.0;

  int eaidx = ea_first_column[bid * addr_stride];
  int daidx = da_first_column[bid * addr_stride];

  for (int i = tx; i < m; i += blockDim.x)
  {
    for (int j = ty; j < n; j += blockDim.y)
    {
      auto idx = eaidx * m * n + i * n + j;
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
    auto eaerror = sum[0] / float(m * n);
    atomicAdd(&out[daidx], eaerror);
  }
}

/*************** class implementation ***************/

RealspaceError::RealspaceError() : CudaFunction("realspace_error") {}

void RealspaceError::setParameters(
    int i, int m, int n, int addr_len, int outlen)
{
  i_ = i;
  m_ = m;
  n_ = n;
  addr_len_ = addr_len;
  outlen_ = outlen;
}

void RealspaceError::setAddrStride(int stride) { addr_stride_ = stride; }

void RealspaceError::setDeviceBuffers(complex<float> *d_difference,
                                      int *d_ea_first_column,
                                      int *d_da_first_column,
                                      float *d_out)
{
  d_difference_ = d_difference;
  d_ea_first_column_ = d_ea_first_column;
  d_da_first_column_ = d_da_first_column;
  d_out_ = d_out;
}

void RealspaceError::allocate()
{
  ScopedTimer t(this, "allocate");
  d_difference_.allocate(i_ * m_ * n_);
  d_ea_first_column_.allocate(addr_len_ * addr_stride_);
  d_da_first_column_.allocate(addr_len_ * addr_stride_);
  d_out_.allocate(outlen_);
}

void RealspaceError::updateErrorOutput(float *d_out) 
{ 
  d_out_ = d_out; 
}

void RealspaceError::transfer_in(const complex<float> *difference,
                                 const int *ea_first_column,
                                 const int *da_first_column)
{
  ScopedTimer t(this, "transfer in");
  gpu_memcpy_h2d(d_difference_.get(), difference, i_ * n_ * m_);
  gpu_memcpy_h2d(
      d_ea_first_column_.get(), ea_first_column, addr_len_ * addr_stride_);
  gpu_memcpy_h2d(
      d_da_first_column_.get(), da_first_column, addr_len_ * addr_stride_);
}

void RealspaceError::run()
{
  ScopedTimer t(this, "run");

  checkCudaErrors(cudaMemset(d_out_.get(), 0, outlen_ * sizeof(*d_out_.get())));

  // always use a 32x32 block of threads
  dim3 threadsPerBlock = {32u, 32u, 1u};
  dim3 blocks = {unsigned(addr_len_), 1u, 1u};
  realspace_error_kernel<<<blocks, threadsPerBlock, 32 * 32 * sizeof(float)>>>(
      d_difference_.get(),
      d_ea_first_column_.get(),
      d_da_first_column_.get(),
      addr_stride_,
      d_out_.get(),
      m_,
      n_);
  checkLaunchErrors();

  timing_sync();
}

void RealspaceError::transfer_out(float *out)
{
  ScopedTimer t(this, "transfer out");
  gpu_memcpy_d2h(out, d_out_.get(), outlen_);
}

/************** interface function ***************/

extern "C" void realspace_error_c(const float *difference,
                                  const int *ea_first_column,
                                  const int *da_first_column,
                                  int addr_len,
                                  float *out,
                                  int i,
                                  int m,
                                  int n,
                                  int outlen)
{
  auto rse = gpuManager.get_cuda_function<RealspaceError>(
      "realspace_error", i, m, n, addr_len, outlen);
  rse->allocate();
  rse->transfer_in(reinterpret_cast<const complex<float> *>(difference),
                   ea_first_column,
                   da_first_column);
  rse->run();
  rse->transfer_out(out);
}