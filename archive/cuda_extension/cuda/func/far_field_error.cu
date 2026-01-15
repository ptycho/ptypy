#include "far_field_error.h"

#include "utils/GpuManager.h"
#include "utils/ScopedTimer.h"

/************* Kernels **********************/

template <int BlockX, int BlockY>
__global__ void far_field_error_kernel(const float *current,
                                       const float *measured,
                                       const unsigned char *mask,
                                       float *out,
                                       int m,
                                       int n)
{
  int batch = blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  extern __shared__ float sum_v[];
  auto sum_mask = (int *)(sum_v + BlockX * BlockY);
  auto shidx = tx * BlockY + ty;
  sum_v[shidx] = 0.0;
  sum_mask[shidx] = 0.0;

  auto offset = batch * m * n;
#pragma unroll(2)
  for (int i = tx; i < m; i += BlockX)
  {
#pragma unroll(1)
    for (int j = ty; j < n; j += BlockY)
    {
      auto idx = offset + i * n + j;
      if (mask[idx])
      {
        auto fdev = current[idx] - measured[idx];
        auto fdev2 = fdev * fdev;
        sum_v[shidx] += fdev2;
        sum_mask[shidx] += 1;
      }
    }
  }

  // now sum up the data in shared memory, tree type reduction
  __syncthreads();
  int nt = BlockX * BlockY;
  int c = nt;
  while (c > 1)
  {
    int half = c / 2;
    if (shidx < half)
    {
      sum_v[shidx] += sum_v[c - shidx - 1];
      sum_mask[shidx] += sum_mask[c - shidx - 1];
    }
    __syncthreads();
    c = c - half;
  }

  if (shidx == 0)
  {
    out[batch] = sum_v[0] / float(sum_mask[0]);
  }
}

/************* class implementation ********************/

FarFieldError::FarFieldError() : CudaFunction("far_field_error") {}

void FarFieldError::setParameters(int i, int m, int n)
{
  i_ = i;
  m_ = m;
  n_ = n;
}

void FarFieldError::setDeviceBuffers(float *d_current,
                                     float *d_measured,
                                     unsigned char *d_mask,
                                     float *d_out)
{
  d_current_ = d_current;
  d_measured_ = d_measured;
  d_mask_ = d_mask;
  d_out_ = d_out;
}

void FarFieldError::allocate()
{
  ScopedTimer t(this, "allocate");
  d_current_.allocate(i_ * m_ * n_);
  d_measured_.allocate(i_ * m_ * n_);
  d_mask_.allocate(i_ * m_ * n_);
  d_out_.allocate(i_);
}

void FarFieldError::updateErrorOutput(float *d_out) { d_out_ = d_out; }

float *FarFieldError::getOutput() const { return d_out_.get(); }

void FarFieldError::transfer_in(const float *current,
                                const float *measured,
                                const unsigned char *mask)
{
  ScopedTimer t(this, "transfer in");
  gpu_memcpy_h2d(d_current_.get(), current, i_ * m_ * n_);
  gpu_memcpy_h2d(d_measured_.get(), measured, i_ * m_ * n_);
  gpu_memcpy_h2d(d_mask_.get(), mask, i_ * m_ * n_);
}

void FarFieldError::run()
{
  ScopedTimer t(this, "run");

  // always use a 32x32 block of threads
  dim3 threadsPerBlock = {32u, 32u, 1u};
  dim3 blocks = {unsigned(i_), 1u, 1u};

  far_field_error_kernel<32, 32>
      <<<blocks, threadsPerBlock, 2 * 32 * 32 * sizeof(float)>>>(
          d_current_.get(),
          d_measured_.get(),
          d_mask_.get(),
          d_out_.get(),
          m_,
          n_);
  checkLaunchErrors();

  timing_sync();
}

void FarFieldError::transfer_out(float *out)
{
  ScopedTimer t(this, "transfer out");
  gpu_memcpy_d2h(out, d_out_.get(), i_);
}

/************* interface function **********************/

extern "C" void far_field_error_c(const float *current,
                                  const float *measured,
                                  const unsigned char *mask,
                                  float *out,
                                  int i,
                                  int m,
                                  int n)
{
  auto ffe =
      gpuManager.get_cuda_function<FarFieldError>("farfield_error", i, m, n);
  ffe->allocate();
  ffe->transfer_in(current, measured, mask);
  ffe->run();
  ffe->transfer_out(out);
}
