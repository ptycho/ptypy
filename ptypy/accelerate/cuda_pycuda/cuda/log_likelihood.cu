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
    log_likelihood(int nmodes,
                  complex<float> *aux,
                  const float *fmask,
                  const float *fmag,
                  const int *addr,
                  float *llerr,
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
  float norm = A * B;

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
      auto I = fmag[a * B + b] * fmag[a * B + b];
      llerr[a * B + b] = fmask[a * B + b] * (acc - I) * (acc - I) / (I + 1) / norm;
    }
  }
}
