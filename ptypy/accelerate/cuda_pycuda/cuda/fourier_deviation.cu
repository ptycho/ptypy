/** fourier_deviation.
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

// version where input is complex and we need to calculate intensity
inline __device__ MATH_TYPE intensity(complex<MATH_TYPE> f)
{
  MATH_TYPE abst = abs(f);
  return abst * abst;  // if we do this manually (real*real +imag*imag)
                       // we get differences to numpy due to rounding
}

// version where input is already an intensity - just return
inline __device__ MATH_TYPE intensity(MATH_TYPE f)
{
  return f;
}

template <class AUX_T>
inline __device__ void fourier_deviation_impl(int nmodes,
                      const AUX_T *f,
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
      acc += intensity(f[a * B + b + idx * A * B]);
    }
    auto fdevv = sqrt(acc) - MATH_TYPE(fmag[a * B + b]);
    fdev[a * B + b] = fdevv;
  }
}

// two kernels - one if input is already an intensity and one where it's not

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
  fourier_deviation_impl(nmodes, f, fmag, fdev, addr, A, B);
}


extern "C" __global__ void __launch_bounds__(1024, 2)
    fourier_deviation_auxintensity(int nmodes,
                      const IN_TYPE *f,
                      const IN_TYPE *fmag,
                      OUT_TYPE *fdev,
                      const int *addr,
                      int A,
                      int B)
{
  fourier_deviation_impl(nmodes, f, fmag, fdev, addr, A, B);
}