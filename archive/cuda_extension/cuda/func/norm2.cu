
#include "norm2.h"
#include "utils/Complex.h"
#include "utils/FinalSumKernel.h"
#include "utils/GpuManager.h"
#include "utils/ScopedTimer.h"

#include <cmath>
#include <cstdlib>

/************ Kernels ***************/

__device__ inline float dev_abs2(float x) { return x * x; }
__device__ inline float dev_abs2(complex<float> x)
{
  return x.real() * x.real() + x.imag() * x.imag();
}

template <class T>
__global__ void norm2_kernel(const T* input, float* output, int size)
{
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;
  extern __shared__ float sum[];

  using std::abs;

  if (gid < size)
  {
    sum[tid] = dev_abs2(input[gid]);
  }
  else
  {
    sum[tid] = 0.0f;
  }

  // now add up sumbuffer in shared memory
  __syncthreads();
  int nt = blockDim.x;
  int c = nt;
  while (c > 1)
  {
    int half = c / 2;
    if (tid < half)
    {
      sum[tid] += sum[c - tid - 1];
    }
    __syncthreads();
    c = c - half;
  }

  if (tid == 0)
  {
    output[blockIdx.x] = sum[0];
  }
}

/************ Class implementation **********/

template <class T>
Norm2<T>::Norm2() : CudaFunction("norm2")
{
}

template <class T>
void Norm2<T>::setParameters(int size)
{
  size_ = size;
  numblocks_ = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

template <class T>
void Norm2<T>::setDeviceBuffers(T* d_input, float* d_output)
{
  d_input_ = d_input;
  d_output_ = d_output;
}

template <class T>
void Norm2<T>::allocate()
{
  ScopedTimer t(this, "allocate");
  d_input_.allocate(size_);
  if (numblocks_ > 1)
  {
    d_intermediate_.allocate(numblocks_);
  }
  d_output_.allocate(1);
}

template <class T>
float* Norm2<T>::getOutput() const
{
  return d_output_.get();
}

template <class T>
void Norm2<T>::transfer_in(const T* input)
{
  ScopedTimer t(this, "transfer in");
  gpu_memcpy_h2d(d_input_.get(), input, size_);
}

template <class T>
void Norm2<T>::run()
{
  ScopedTimer t(this, "run");
  if (numblocks_ == 1)
  {
    norm2_kernel<<<numblocks_, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        d_input_.get(), d_output_.get(), size_);
  }
  else
  {
    norm2_kernel<<<numblocks_, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        d_input_.get(), d_intermediate_.get(), size_);
    // now we have one output per block, so a final kernel is needed
    int nthreads = numblocks_;
    if (nthreads > 1024)
      nthreads = 1024;
    final_sum<<<1, nthreads, nthreads * sizeof(float)>>>(
        d_intermediate_.get(), d_output_.get(), numblocks_);
  }
  timing_sync();
}

template <class T>
void Norm2<T>::transfer_out(float* output)
{
  ScopedTimer t(this, "transfer out");
  gpu_memcpy_d2h(output, d_output_.get(), 1);
}

/*********** interface function *************/

template <class T>
void norm2_tc(const T* data, float* out, int size)
{
  auto n2 = gpuManager.get_cuda_function<Norm2<T>>(
      "norm2<" + getTypeName<T>() + ">", size);
  n2->allocate();
  n2->transfer_in(data);
  n2->run();
  n2->transfer_out(out);
}

template class Norm2<float>;
template class Norm2<complex<float>>;

extern "C" void norm2_c(const float* data, float* out, int size, int isComplex)
{
  if (isComplex != 0)
  {
    norm2_tc(reinterpret_cast<const complex<float>*>(data), out, size);
  }
  else
  {
    norm2_tc(data, out, size);
  }
}