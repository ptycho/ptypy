#include <thrust/complex.h>
using thrust::complex;

extern "C" __global__ void build_aux_no_ex(CTYPE* auxilliary_wave,
                                           int aRows,
                                           int aCols,
                                           const CTYPE* __restrict__ probe,
                                           int pRows,
                                           int pCols,
                                           const CTYPE* __restrict__ obj,
                                           int oRows,
                                           int oCols,
                                           const int* __restrict__ addr,
                                           FTYPE fac,
                                           int doAdd)
{
  int bid = blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  const int addr_stride = 15;

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
      auto tmp = obj[b * oCols + c] * probe[b * pCols + c] * fac;
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