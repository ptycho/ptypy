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
__global__ void ob_update(
    const complex<float>* exit_wave,
    int A,
    int B,
    int C,
    const complex<float>* __restrict__ probe,
    int D,
    int E,
    int F,
    complex<float>* obj,
    int G,
    int H,
    int I,
    const int* addr,
    complex<float>* denominator
    )
    {
      int bid = blockIdx.x;
      int tx = threadIdx.x;
      int ty = threadIdx.y;
      int addr_stride = 15;

      const int* oa = addr + 3 + bid * addr_stride;
      const int* pa = addr + bid * addr_stride;
      const int* ea = addr + 6 + bid * addr_stride;

      probe += pa[0] * E * F + pa[1] * F + pa[2];
      obj += oa[0] * H * I + oa[1] * I + oa[2];
      denominator += oa[0] * H * I + oa[1] * I + oa[2];

      assert(oa[0] * H * I + oa[1] * I + oa[2] + (B - 1) * I + C - 1 < G * H * I);

      exit_wave += ea[0] * B * C;

      for (int b = tx; b < B; b += blockDim.x)
      {
        for (int c = ty; c < C; c += blockDim.y)
        {
          atomicAdd(&obj[b * I + c], conj(probe[b * F + c]) * exit_wave[b * C + c] );
          atomicAdd(&denominator[b * I + c], probe[b * F + c] * conj(probe[b * F + c]) );
          }
       }
}

}