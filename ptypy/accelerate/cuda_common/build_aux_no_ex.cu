/** build_aux without exit wave kernel.
 *
 * Data types:
 * - IN_TYPE: the data type for the inputs (float or double)
 * - OUT_TYPE: the data type for the outputs (float or double - for aux wave)
 * - MATH_TYPE: the data type used for computation 
 */

#include "common.cuh"

extern "C" __global__ void build_aux_no_ex(complex<OUT_TYPE>* auxilliary_wave,
                                           int aRows,
                                           int aCols,
                                           const complex<IN_TYPE>* __restrict__ probe,
                                           int pRows,
                                           int pCols,
                                           const complex<IN_TYPE>* __restrict__ obj,
                                           int oRows,
                                           int oCols,
                                           const int* __restrict__ addr,
                                           IN_TYPE fac_,
                                           int doAdd)
{
  int bid = blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  const int addr_stride = 15;
  const MATH_TYPE fac = fac_;   // type conversion

  const int* oa = addr + 3 + bid * addr_stride;
  const int* pa = addr + bid * addr_stride;
  const int* ea = addr + 6 + bid * addr_stride;

  obj += oa[0] * oRows * oCols + oa[1] * oCols + oa[2];
  probe += pa[0] * pRows * pCols + pa[1] * pCols + pa[2];
  auxilliary_wave += ea[0] * aRows * aCols;

  for (int b = ty; b < aRows; b += blockDim.y)
  {
#   pragma unroll(4)
    for (int c = tx; c < aCols; c += blockDim.x)
    {
      complex<MATH_TYPE> t_obj = obj[b * oCols + c];
      complex<MATH_TYPE> t_probe = probe[b * pCols + c];
      auto tmp = t_obj * t_probe * fac;
      if (doAdd)
      {
        auxilliary_wave[b * aCols + c] += tmp;
      }
      else
      {
        auxilliary_wave[b * aCols + c] = tmp;
      }
    }
  }
}

extern "C" __global__ void build_aux2_no_ex(complex<OUT_TYPE>* auxilliary_wave,
                                           int aRows,
                                           int aCols,
                                           const complex<IN_TYPE>* __restrict__ probe,
                                           int pRows,
                                           int pCols,
                                           const complex<IN_TYPE>* __restrict__ obj,
                                           int oRows,
                                           int oCols,
                                           const int* __restrict__ addr,
                                           IN_TYPE fac_,
                                           int doAdd)
{
  int bid = blockIdx.z;
  int tx = threadIdx.x;
  int b = threadIdx.y + blockIdx.y * blockDim.y;
  if (b >= aRows)
    return;
  const int addr_stride = 15;
  const MATH_TYPE fac = fac_;   // type conversion

  const int* oa = addr + 3 + bid * addr_stride;
  const int* pa = addr + bid * addr_stride;
  const int* ea = addr + 6 + bid * addr_stride;

  obj += oa[0] * oRows * oCols + oa[1] * oCols + oa[2];
  probe += pa[0] * pRows * pCols + pa[1] * pCols + pa[2];
  auxilliary_wave += ea[0] * aRows * aCols;

  for (int c = tx; c < aCols; c += blockDim.x)
  {
    complex<MATH_TYPE> t_obj = obj[b * oCols + c];
    complex<MATH_TYPE> t_probe = probe[b * pCols + c];
    auto tmp = t_obj * t_probe * fac;
    if (doAdd)
    {
      auxilliary_wave[b * aCols + c] += tmp;
    }
    else
    {
      auxilliary_wave[b * aCols + c] = tmp;
    }
  }

}