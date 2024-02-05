#include "common.cuh"

template <class T>
__device__ inline T dotmul(const T& a, const T& b)
{
  return a * b;
}

template <class T>
__device__ inline T dotmul(const complex<T>& a, const complex<T>& b)
{
  //return (a * conj(b)).real();
  return a.real() * b.real() + a.imag() * b.imag();
}

extern "C" __global__ void dot(const IN_TYPE* a,
                               const IN_TYPE* b,
                               int size,
                               ACC_TYPE* out)
{
  int tx = threadIdx.x;
  int ix = tx + blockIdx.x * blockDim.x;

  __shared__ ACC_TYPE sh[1024];

  if (ix < size)
  {
    sh[tx] = dotmul(a[ix], b[ix]);
  }
  else
  {
    sh[tx] = ACC_TYPE(0);
  }
  __syncthreads();

  int nt = blockDim.x;
  int c = nt;

  while (c > 1)
  {
    int half = c / 2;
    if (tx < half)
    {
      sh[tx] += sh[c - tx - 1];
    }
    __syncthreads();
    c = c - half;
  }

  if (tx == 0 && ix < size)
  {
    out[blockIdx.x] = sh[0];
  }
}