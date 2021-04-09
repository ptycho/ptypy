/** fourier_error.
 *
 * Data types:
 * - IN_TYPE: the data type for the inputs (float or double)
 * - OUT_TYPE: the data type for the outputs (float or double)
 * - MATH_TYPE: the data type used for computation 
 */


#include <cassert>
#include <cmath>
#include <thrust/complex.h>
using std::sqrt;
using thrust::abs;
using thrust::complex;

// version if input is complex, i.e. we need to calc abs(.)**2
inline __device__ MATH_TYPE aux_intensity(complex<MATH_TYPE> aux_t) {
  MATH_TYPE abst = abs(aux_t);
  return abst * abst; // if we do this manually (real*real +imag*imag)
                      // we get differences to numpy due to rounding
}

// version if input is real, so we can just return it
inline __device__ MATH_TYPE aux_intensity(MATH_TYPE aux_t) {
  return aux_t;
}

template <class AUX_T>
inline __device__ void fourier_error_impl(
                  int nmodes,
                  const AUX_T *f,
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
        acc += aux_intensity(f[a * B + b + idx * A * B]);
      }
      auto fdevv = sqrt(acc) - MATH_TYPE(fmag[a * B + b]);
      ferr[a * B + b] = (fmask[a * B + b] * fdevv * fdevv) / mask_sum[ma[0]];
      fdev[a * B + b] = fdevv;
    }
  }
}

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
  fourier_error_impl(nmodes, f, fmask, fmag, fdev, ferr, mask_sum, addr, A, B);
}


extern "C" __global__ void __launch_bounds__(1024, 2)
    fourier_error_auxintensity(int nmodes,
                  const IN_TYPE *f,
                  const IN_TYPE *fmask,
                  const IN_TYPE *fmag,
                  OUT_TYPE *fdev,
                  OUT_TYPE *ferr,
                  const IN_TYPE *mask_sum,
                  const int *addr,
                  int A,
                  int B)
{
  fourier_error_impl(nmodes, f, fmask, fmag, fdev, ferr, mask_sum, addr, A, B);
}
