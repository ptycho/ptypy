/** fourier_error.
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
    fourier_error(int nmodes,
                  const complex<IN_TYPE> *f,
                  const IN_TYPE *fmask,
                  const IN_TYPE *fmag,
                  OUT_TYPE *fdev,
                  OUT_TYPE *ferr,
                  const IN_TYPE *mask_sum,
                  const int *addr,
                  int A,
                  int B)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int addr_stride = 15;

  const int *ea = addr + 6 + (blockIdx.x * nmodes) * addr_stride;
  const int *da = addr + 9 + (blockIdx.x * nmodes) * addr_stride;
  const int *ma = addr + 12 + (blockIdx.x * nmodes) * addr_stride;

  f += ea[0] * A * B;
  fdev += da[0] * A * B;
  fmag += da[0] * A * B;
  fmask += ma[0] * A * B;
  ferr += da[0] * A * B;

  for (int a = ty; a < A; a += blockDim.y)
  {
    for (int b = tx; b < B; b += blockDim.x)
    {
      MATH_TYPE acc = MATH_TYPE(0);
      for (int idx = 0; idx < nmodes; ++idx)
      {
        complex<MATH_TYPE> t_f = f[a * B + b + idx * A * B];
        MATH_TYPE abs_exit_wave = abs(t_f);
        acc += abs_exit_wave *
               abs_exit_wave;  // if we do this manually (real*real +imag*imag)
                               // we get differences to numpy due to rounding
      }
      auto fdevv = sqrt(acc) - MATH_TYPE(fmag[a * B + b]);
      ferr[a * B + b] = (fmask[a * B + b] * fdevv * fdevv) / mask_sum[ma[0]];
      fdev[a * B + b] = fdevv;
    }
  }
}
