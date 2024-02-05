#include "common.cuh"

// specify max number of threads/block and min number of blocks per SM,
// to assist the compiler in register optimisations.
// We achieve a higher occupancy in this case, as less registers are used
// (guided by profiler)
extern "C" __global__ void __launch_bounds__(1024, 2)
    exit_error(int nmodes,
               const complex<IN_TYPE> * __restrict aux,
               OUT_TYPE *ferr,
               const int * __restrict addr,
               int A,
               int B)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int addr_stride = 15;
  MATH_TYPE denom = A * B;

  const int *ea = addr + 6 + (blockIdx.x * nmodes) * addr_stride;
  const int *da = addr + 9 + (blockIdx.x * nmodes) * addr_stride;

  aux += ea[0] * A * B;
  ferr += da[0] * A * B;

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
      ferr[a * B + b] = OUT_TYPE(acc / denom);
    }
  }
}
