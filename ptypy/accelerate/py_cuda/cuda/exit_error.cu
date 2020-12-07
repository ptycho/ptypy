#include <cassert>
#include <cmath>
#include <thrust/complex.h>
using std::sqrt;
using thrust::abs;
using thrust::complex;

// specify max number of threads/block and min number of blocks per SM,
// to assist the compiler in register optimisations.
// We achieve a higher occupancy in this case, as less registers are used
// (guided by profiler)
extern "C" __global__ void __launch_bounds__(1024, 2)
    exit_error(int nmodes,
               complex<float> *aux,
               float *ferr,
               const int *addr,
               int A,
               int B)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int addr_stride = 15;
  float denom = A * B;

  const int *ea = addr + 6 + (blockIdx.x * nmodes) * addr_stride;
  const int *da = addr + 9 + (blockIdx.x * nmodes) * addr_stride;

  aux += ea[0] * A * B;
  ferr += da[0] * A * B;

  for (int a = ty; a < A; a += blockDim.y)
  {
    for (int b = tx; b < B; b += blockDim.x)
    {
      float acc = 0.0;
      for (int idx = 0; idx < nmodes; ++idx)
      {
        float abs_exit_wave = abs(aux[a * B + b + idx * A * B]);
        acc += abs_exit_wave *
               abs_exit_wave;  // if we do this manually (real*real +imag*imag)
                               // we get differences to numpy due to rounding
      }
      ferr[a * B + b] = acc / denom;
    }
  }
}
