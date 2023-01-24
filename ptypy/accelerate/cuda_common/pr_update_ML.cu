/** pr_update_ML.
 *
 * Data types:
 * - IN_TYPE: the data type for the inputs (float or double)
 * - OUT_TYPE: the data type for the outputs (float or double)
 * - MATH_TYPE: the data type used for computation
 */


#include "common.cuh"

template <class T>
__device__ inline void atomicAdd(complex<T>* x, const complex<T>& y)
{
  auto xf = reinterpret_cast<T*>(x);
  atomicAdd(xf, y.real());
  atomicAdd(xf + 1, y.imag());
}

extern "C" __global__ void pr_update_ML(const complex<IN_TYPE>* __restrict__ exit_wave,
                                        int A,
                                        int B,
                                        int C,
                                        complex<OUT_TYPE>* probe,
                                        int D,
                                        int E,
                                        int F,
                                        const complex<IN_TYPE>* __restrict__ obj,
                                        int G,
                                        int H,
                                        int I,
                                        const int* __restrict__ addr,
                                        IN_TYPE fac_)
{
  assert(B == E);  // prsh[1]
  assert(C == F);  // prsh[2]
  const int bid = blockIdx.x;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int addr_stride = 15;
  MATH_TYPE fac = fac_;

  const int* oa = addr + 3 + bid * addr_stride;
  const int* pa = addr + bid * addr_stride;
  const int* ea = addr + 6 + bid * addr_stride;

  probe += pa[0] * E * F + pa[1] * F + pa[2];
  obj += oa[0] * H * I + oa[1] * I + oa[2];

  assert(oa[0] * H * I + oa[1] * I + oa[2] + (B - 1) * I + C - 1 < G * H * I);

  exit_wave += ea[0] * B * C;

  for (int b = ty; b < B; b += blockDim.y)
  {
    for (int c = tx; c < C; c += blockDim.x)
    {
      complex<MATH_TYPE> obj_val = obj[b * I + c];
      complex<MATH_TYPE> exit_val = exit_wave[b * C + c];
      complex<MATH_TYPE> add_val_m = conj(obj_val) * exit_val * fac;
      complex<OUT_TYPE> add_val = add_val_m;
      atomicAdd(&probe[b * F + c], add_val);
    }
  }
}
