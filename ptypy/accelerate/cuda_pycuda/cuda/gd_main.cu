#include <thrust/complex.h>
using thrust::complex;

extern "C" __global__ void gd_main(const FTYPE* Imodel,
                                   const FTYPE* I,
                                   const FTYPE* w,
                                   FTYPE* err,
                                   CTYPE* aux,
                                   int z,
                                   int modes,
                                   int x)
{
  int iz = blockIdx.z;
  int ix = threadIdx.x + blockIdx.x * blockDim.x;

  if (iz >= z || ix >= x)
    return;

  auto DI = Imodel[iz * x + ix] - I[iz * x + ix];
  auto tmp = w[iz * x + ix] * DI;
  err[iz * x + ix] = tmp * DI;

  // now set this for all modes (promote)
  for (int m = 0; m < modes; ++m)
  {
    aux[iz * x * modes + m * x + ix] *= tmp;
  }
}