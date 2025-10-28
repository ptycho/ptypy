/** build_aux for position correction.
 *
 * Data types:
 * - IN_TYPE: the data type for the inputs (float or double)
 * - OUT_TYPE: the data type for the outputs (float or double - for aux wave)
 * - MATH_TYPE: the data type used for computation 
 */

#include "common.cuh"

extern "C" __global__ void build_aux_position_correction(
    complex<OUT_TYPE>* auxiliary_wave,
    const complex<IN_TYPE>* __restrict__ probe,
    int B,
    int C,
    const complex<IN_TYPE>* __restrict__ obj,
    int H,
    int I,
    const int* __restrict__ addr)
{
  int bid = blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int addr_stride = 15;

  const int* oa = addr + 3 + bid * addr_stride;
  const int* pa = addr + bid * addr_stride;
  const int* ea = addr + 6 + bid * addr_stride;

  probe += pa[0] * B * C + pa[1] * C + pa[2];
  obj += oa[0] * H * I + oa[1] * I + oa[2];
  auxiliary_wave += ea[0] * B * C;

  for (int b = ty; b < B; b += blockDim.y)
  {
#pragma unroll(4)  // we use blockDim.x = 32, and C is typically more than 128
                   // (it will work for less as well)
    for (int c = tx; c < C; c += blockDim.x)
    {
      complex<MATH_TYPE> t_obj = obj[b * I + c];
      complex<MATH_TYPE> t_probe = probe[b * C + c];
      auxiliary_wave[b * C + c] = t_obj * t_probe;
    }
  }
}
