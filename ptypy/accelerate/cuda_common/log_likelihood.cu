/** log_likelihood kernel.
 *
 * Data types:
 * - IN_TYPE: the data type for the inputs (float or double)
 * - OUT_TYPE: the data type for the outputs (float or double)
 * - MATH_TYPE: the data type used for computation
 */

#include "common.cuh"

// specify max number of threads/block and min number of blocks per SM,
// to assist the compiler in register optimisations.
// We achieve a higher occupancy in this case, as less registers are used
// (guided by profiler)
extern "C" __global__ void __launch_bounds__(1024, 2)
    log_likelihood(int nmodes,
                   complex<OUT_TYPE> *aux,
                   const IN_TYPE *fmask,
                   const IN_TYPE *fmag,
                   const int *addr,
                   IN_TYPE *llerr,
                   int A,
                   int B)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int addr_stride = 15;

  const int *ea = addr + 6 + (blockIdx.x * nmodes) * addr_stride;
  const int *da = addr + 9 + (blockIdx.x * nmodes) * addr_stride;
  const int *ma = addr + 12 + (blockIdx.x * nmodes) * addr_stride;

  aux += ea[0] * A * B;
  fmag += da[0] * A * B;
  fmask += ma[0] * A * B;
  llerr += da[0] * A * B;
  MATH_TYPE norm = A * B;

  for (int a = ty; a < A; a += blockDim.y)
  {
    for (int b = tx; b < B; b += blockDim.x)
    {
      MATH_TYPE acc = 0.0;
      for (int idx = 0; idx < nmodes; ++idx)
      {
        complex<MATH_TYPE> t_aux = aux[a * B + b + idx * A * B];
        MATH_TYPE abs_exit_wave = abs(t_aux);
        acc += abs_exit_wave *
               abs_exit_wave;  // if we do this manually (real*real +imag*imag)
                               // we get differences to numpy due to rounding
      }
      auto I = MATH_TYPE(fmag[a * B + b]) * MATH_TYPE(fmag[a * B + b]);
      llerr[a * B + b] =
          MATH_TYPE(fmask[a * B + b]) * (acc - I) * (acc - I) / (I + 1) / norm;
    }
  }
}


extern "C" __global__ void 
    log_likelihood2(int nmodes,
                   complex<OUT_TYPE> *aux,
                   const IN_TYPE *fmask,
                   const IN_TYPE *fmag,
                   const int *addr,
                   IN_TYPE *llerr,
                   int A,
                   int B)
{
  int bid = blockIdx.z;
  int tx = threadIdx.x;
  int a = threadIdx.y + blockIdx.y * blockDim.y;
  if (a >= A)
    return;
  int addr_stride = 15;

  const int *ea = addr + 6 + (bid * nmodes) * addr_stride;
  const int *da = addr + 9 + (bid * nmodes) * addr_stride;
  const int *ma = addr + 12 + (bid * nmodes) * addr_stride;

  aux += ea[0] * A * B;
  fmag += da[0] * A * B;
  fmask += ma[0] * A * B;
  llerr += da[0] * A * B;
  MATH_TYPE norm = A * B;

  for (int b = tx; b < B; b += blockDim.x)
  {
    MATH_TYPE acc = 0.0;
    for (int idx = 0; idx < nmodes; ++idx)
    {
      complex<MATH_TYPE> t_aux = aux[a * B + b + idx * A * B];
      MATH_TYPE abs_exit_wave = abs(t_aux);
      acc += abs_exit_wave *
              abs_exit_wave;  // if we do this manually (real*real +imag*imag)
                              // we get differences to numpy due to rounding
    }
    auto I = MATH_TYPE(fmag[a * B + b]) * MATH_TYPE(fmag[a * B + b]);
    llerr[a * B + b] =
        MATH_TYPE(fmask[a * B + b]) * (acc - I) * (acc - I) / (I + 1) / norm;
  }
}

// ML variant which uses weights and intensity directly.
// Based of log_likelihood
extern "C" __global__ void __launch_bounds__(1024, 2)
    log_likelihood_ml(int nmodes,
                   complex<OUT_TYPE> *aux,
                   const IN_TYPE *weights,
                   const IN_TYPE *I,
                   const int *addr,
                   IN_TYPE *llerr,
                   int A,
                   int B)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int addr_stride = 15;

  const int *ea = addr + 6 + (blockIdx.x * nmodes) * addr_stride;
  const int *da = addr + 9 + (blockIdx.x * nmodes) * addr_stride;
  const int *ma = addr + 12 + (blockIdx.x * nmodes) * addr_stride;

  aux += ea[0] * A * B;
  I += da[0] * A * B;
  weights += ma[0] * A * B;
  llerr += da[0] * A * B;
  MATH_TYPE norm = A * B;

  for (int a = ty; a < A; a += blockDim.y)
  {
    for (int b = tx; b < B; b += blockDim.x)
    {
      MATH_TYPE acc = 0.0;
      MATH_TYPE i = I[a * B + b];
      for (int idx = 0; idx < nmodes; ++idx)
      {
        complex<MATH_TYPE> t_aux = aux[a * B + b + idx * A * B];
        MATH_TYPE abs_exit_wave = abs(t_aux);
        acc += abs_exit_wave *
               abs_exit_wave;  // if we do this manually (real*real +imag*imag)
                               // we get differences to numpy due to rounding
      }
      llerr[a * B + b] =
          MATH_TYPE(weights[a * B + b]) * (acc - i) * (acc - i) / norm;
    }
  }
}