/** build_aux kernel.
 *
 * Data types:
 * - IN_TYPE: the data type for the inputs (float or double)
 * - OUT_TYPE: the data type for the outputs (float or double - for aux wave)
 * - MATH_TYPE: the data type used for computation
 */

#include <thrust/complex.h>
using thrust::complex;

// core calculation function - used by both kernels and inlined
inline __device__ complex<MATH_TYPE> calculate(
    const complex<MATH_TYPE>& t_obj,
    const complex<MATH_TYPE>& t_probe,
    const complex<MATH_TYPE>& t_ex,
    MATH_TYPE alpha)
{
  return t_obj * t_probe * (MATH_TYPE(1) + alpha) - t_ex * alpha;
}

extern "C" __global__ void build_aux(
    complex<OUT_TYPE>* auxiliary_wave,
    const complex<IN_TYPE>* __restrict__ exit_wave,
    int B,
    int C,
    const complex<IN_TYPE>* __restrict__ probe,
    int E,
    int F,
    const complex<IN_TYPE>* __restrict__ obj,
    int H,
    int I,
    const int* __restrict__ addr,
    IN_TYPE alpha_)
{
  int bid = blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int addr_stride = 15;
  const MATH_TYPE alpha = alpha_;  // type conversion

  const int* oa = addr + 3 + bid * addr_stride;
  const int* pa = addr + bid * addr_stride;
  const int* ea = addr + 6 + bid * addr_stride;

  probe += pa[0] * E * F + pa[1] * F + pa[2];
  obj += oa[0] * H * I + oa[1] * I + oa[2];
  exit_wave += ea[0] * B * C;
  auxiliary_wave += ea[0] * B * C;

  for (int b = ty; b < B; b += blockDim.y)
  {
#pragma unroll(4)  // we use blockDim.x = 32, and C is typically more than 128
                   // (it will work for less as well)
    for (int c = tx; c < C; c += blockDim.x)
    {
      auxiliary_wave[b * C + c] = calculate(
          obj[b * I + c], probe[b * F + c], exit_wave[b * C + c], alpha);
    }
  }
}

extern "C" __global__ void build_aux2(
    complex<OUT_TYPE>* auxiliary_wave,
    const complex<IN_TYPE>* __restrict__ exit_wave,
    int B,
    int C,
    const complex<IN_TYPE>* __restrict__ probe,
    int E,
    int F,
    const complex<IN_TYPE>* __restrict__ obj,
    int H,
    int I,
    const int* __restrict__ addr,
    IN_TYPE alpha_)
{
  int bid = blockIdx.z;
  int tx = threadIdx.x;
  int b = threadIdx.y + blockIdx.y * blockDim.y;
  if (b >= B)
    return;
  int addr_stride = 15;
  const MATH_TYPE alpha = alpha_;  // type conversion

  const int* oa = addr + 3 + bid * addr_stride;
  const int* pa = addr + bid * addr_stride;
  const int* ea = addr + 6 + bid * addr_stride;

  probe += pa[0] * E * F + pa[1] * F + pa[2];
  obj += oa[0] * H * I + oa[1] * I + oa[2];
  exit_wave += ea[0] * B * C;
  auxiliary_wave += ea[0] * B * C;

  for (int c = tx; c < C; c += blockDim.x)
  {
    auxiliary_wave[b * C + c] = calculate(
        obj[b * I + c], probe[b * F + c], exit_wave[b * C + c], alpha);
  }
}
