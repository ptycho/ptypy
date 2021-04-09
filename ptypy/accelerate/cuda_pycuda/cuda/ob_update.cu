#include <thrust/complex.h>
using thrust::complex;

template <class T>
__device__ inline void atomicAdd(complex<T>* x, complex<T> y)
{
  auto xf = reinterpret_cast<T*>(x);
  atomicAdd(xf, y.real());
  atomicAdd(xf + 1, y.imag());
}

extern "C" __global__ void ob_update(
    const complex<float>* __restrict__ exit_wave,
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
    const int* __restrict__ addr,
    DENOM_TYPE* denominator)
{
  const int bid = blockIdx.x;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int addr_stride = 15;

  const int* oa = addr + 3 + bid * addr_stride;
  const int* pa = addr + bid * addr_stride;
  const int* ea = addr + 6 + bid * addr_stride;

  probe += pa[0] * E * F + pa[1] * F + pa[2];
  obj += oa[0] * H * I + oa[1] * I + oa[2];
  denominator += oa[0] * H * I + oa[1] * I + oa[2];

  assert(oa[0] * H * I + oa[1] * I + oa[2] + (B - 1) * I + C - 1 < G * H * I);

  exit_wave += ea[0] * B * C;

  for (int b = ty; b < B; b += blockDim.y)
  {
    for (int c = tx; c < C; c += blockDim.x)
    {
      auto probe_val = probe[b * F + c];
      atomicAdd(&obj[b * I + c], conj(probe_val) * exit_wave[b * C + c]);
      auto denomreal = reinterpret_cast<float*>(&denominator[b * I + c]);
      auto upd_probe = probe_val.real() * probe_val.real() +
                       probe_val.imag() * probe_val.imag();
      atomicAdd(denomreal, upd_probe);
    }
  }
}
