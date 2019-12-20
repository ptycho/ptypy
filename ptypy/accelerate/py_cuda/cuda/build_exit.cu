#include <iostream>
#include <utility>
#include <thrust/complex.h>
using thrust::complex;
__device__ inline void atomicAdd(complex<float>* x, complex<float> y)
    {
      float* xf = reinterpret_cast<float*>(x);
      atomicAdd(xf, y.real());
      atomicAdd(xf + 1, y.imag());
    }

extern "C"{
__global__ void build_exit(
    complex<float>* auxiliary_wave,
    complex<float>* exit_wave,
    int B,
    int C,
    const complex<float>* __restrict__ probe,
    int E,
    int F,
    const complex<float>* __restrict__ obj,
    int H,
    int I,
    const int* __restrict__ addr
    )
    {
      int bid = blockIdx.x;
      int tx = threadIdx.x;
      int ty = threadIdx.y;
      const int addr_stride = 15;

      const int* oa = addr + 3 + bid * addr_stride;
      const int* pa = addr + bid * addr_stride;
      const int* ea = addr + 6 + bid * addr_stride;

      probe += pa[0] * E * F + pa[1] * F + pa[2];
      obj += oa[0] * H * I + oa[1] * I + oa[2];
      exit_wave += ea[0] * B * C;
      auxiliary_wave += ea[0] * B * C;

      for (int b = ty; b < B; b += blockDim.y)
      {
        for (int c = tx; c < C; c += blockDim.x)
        {
          atomicAdd(&auxiliary_wave[b * C + c], -probe[b * F + c] * obj[b * I + c]); // atomicSub is only for ints
          atomicAdd(&exit_wave[b * C + c], auxiliary_wave[b * C + c]);
        }
      }
    }
}