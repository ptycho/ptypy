/** fourier_deviation.
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
    fourier_deviation(int nmodes,
                      const complex<IN_TYPE> *f,
                      const IN_TYPE *fmag,
                      OUT_TYPE *fdev,
                      const int *addr,
                      int A,
                      int B)
{
  const int bid = blockIdx.z;
  const int tx = threadIdx.x;
  const int a = threadIdx.y + blockIdx.y * blockDim.y;
  const int addr_stride = 15;

  const int *ea = addr + 6 + (bid * nmodes) * addr_stride;
  const int *da = addr + 9 + (bid * nmodes) * addr_stride;

  f += ea[0] * A * B;
  fdev += da[0] * A * B;
  fmag += da[0] * A * B;

  if (a >= A)
    return;

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
    fdev[a * B + b] = fdevv;
  }
}
