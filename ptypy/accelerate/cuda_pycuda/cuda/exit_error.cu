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


// generic template that works for complex aux, or intensity aux
template <class AUX_T>
inline __device__ void exit_error_impl(int nmodes,
  const AUX_T* __restrict aux,
  OUT_TYPE *ferr,
  const int* __restrict addr,
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
        acc += aux_intensity(aux[a * B + b + idx * A * B]);
      }
      ferr[a * B + b] = OUT_TYPE(acc / denom);
    }
  }
}

// 2 kernels using the same template above - only difference is the input
// type for aux (and the name)


// specify max number of threads/block and min number of blocks per SM,
// to assist the compiler in register optimisations.
// We achieve a higher occupancy in this case, as less registers are used
// (guided by profiler)
extern "C" __global__ void __launch_bounds__(1024, 2)
    exit_error(int nmodes,
               const complex<IN_TYPE>* __restrict aux,
               OUT_TYPE *ferr,
               const int * __restrict addr,
               int A,
               int B)
{
  exit_error_impl(nmodes, aux, ferr, addr, A, B);
}


extern "C" __global__ void __launch_bounds__(1024, 2)
    exit_error_auxintensity(int nmodes,
               const IN_TYPE* __restrict aux,
               OUT_TYPE *ferr,
               const int * __restrict addr,
               int A,
               int B)
{
  exit_error_impl(nmodes, aux, ferr, addr, A, B);
}