/** build_exit kernel.
 *
 * Data types:
 * - IN_TYPE: the data type for the inputs (float or double)
 * - OUT_TYPE: the data type for the outputs (float or double - for aux wave)
 * - MATH_TYPE: the data type used for computation 
 */


#include "common.cuh"

template <class T>
__device__ inline void atomicAdd(complex<T>* x, complex<T> y)
{
  auto xf = reinterpret_cast<T*>(x);
  atomicAdd(xf, y.real());
  atomicAdd(xf + 1, y.imag());
}

extern "C" __global__ void make_exit(complex<OUT_TYPE>* auxiliary_wave,
                                      complex<OUT_TYPE>* exit_wave,
                                      int B,
                                      int C,
                                      const complex<IN_TYPE>* __restrict__ probe,
                                      int E,
                                      int F,
                                      const complex<IN_TYPE>* __restrict__ obj,
                                      int H,
                                      int I,
                                      const int* __restrict__ addr,
                                      IN_TYPE coeff_a_,
                                      IN_TYPE coeff_po_,
                                      IN_TYPE coeff_e_)
{
  int bid = blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  const int addr_stride = 15;
  const MATH_TYPE coeff_a = coeff_a_;  // type conversion
  const MATH_TYPE coeff_po = coeff_po_;  // type conversion
  const MATH_TYPE coeff_e = coeff_e_;  // type conversion

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
      complex<MATH_TYPE> auxv = auxiliary_wave[b * C + c];
      complex<MATH_TYPE> t_probe = probe[b * F + c];
      complex<MATH_TYPE> t_obj = obj[b * I + c];
      complex<MATH_TYPE> t_exit = exit_wave[b * C + c];
      auxv *= coeff_a;
      auxv += coeff_po * t_probe * t_obj;
      auxv += coeff_e * t_exit;
      exit_wave[b * C + c] += auxv;
      auxiliary_wave[b * C + c] = auxv;
    }
  }
}
