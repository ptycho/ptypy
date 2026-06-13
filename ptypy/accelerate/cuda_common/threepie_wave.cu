/** 3PIE local wave transfer kernels.
 *
 * These kernels move a local propagated wave between full probe storage and
 * the auxiliary wave buffer using ptypy's serialized address layout.
 */

#include "common.cuh"

extern "C" __global__ void threepie_pr_to_aux(complex<OUT_TYPE>* auxilliary_wave,
                                              int aRows,
                                              int aCols,
                                              const complex<IN_TYPE>* __restrict__ probe,
                                              int pRows,
                                              int pCols,
                                              const int* __restrict__ addr)
{
  int bid = blockIdx.z;
  int tx = threadIdx.x;
  int b = threadIdx.y + blockIdx.y * blockDim.y;
  if (b >= aRows)
    return;

  const int addr_stride = 15;
  const int* pa = addr + bid * addr_stride;
  const int* ea = addr + 6 + bid * addr_stride;

  probe += pa[0] * pRows * pCols + (pa[1] + b) * pCols + pa[2];
  auxilliary_wave += ea[0] * aRows * aCols + b * aCols;

  for (int c = tx; c < aCols; c += blockDim.x)
  {
    auxilliary_wave[c] = probe[c];
  }
}

extern "C" __global__ void threepie_aux_to_pr(const complex<IN_TYPE>* __restrict__ auxilliary_wave,
                                              int aRows,
                                              int aCols,
                                              complex<OUT_TYPE>* probe,
                                              int pRows,
                                              int pCols,
                                              const int* __restrict__ addr)
{
  int bid = blockIdx.z;
  int tx = threadIdx.x;
  int b = threadIdx.y + blockIdx.y * blockDim.y;
  if (b >= aRows)
    return;

  const int addr_stride = 15;
  const int* pa = addr + bid * addr_stride;
  const int* ea = addr + 6 + bid * addr_stride;

  auxilliary_wave += ea[0] * aRows * aCols + b * aCols;
  probe += pa[0] * pRows * pCols + (pa[1] + b) * pCols + pa[2];

  for (int c = tx; c < aCols; c += blockDim.x)
  {
    probe[c] = auxilliary_wave[c];
  }
}
