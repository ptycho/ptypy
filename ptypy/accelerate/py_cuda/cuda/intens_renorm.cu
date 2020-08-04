#include <thrust/complex.h>
using thrust::complex;

extern "C" __global__ void step1(const FTYPE* Imodel,
                                   const FTYPE* I,
                                   const FTYPE* w,
                                   FTYPE* num,
                                   FTYPE* den,
                                   int z,
                                   int x)
{
  int iz = blockIdx.z;
  int ix = threadIdx.x + blockIdx.x * blockDim.x;

  if (iz >= z || ix >= x)
    return;

  auto tmp = w[iz * x + ix] * Imodel[iz * x + ix];
  num[iz * x + ix] = tmp * I[iz * x + ix];
  den[iz * x + ix] = tmp * Imodel[iz * x + ix];
}

extern "C" __global__ void step2(const FTYPE* den,
                                 FTYPE* fic,
                                 FTYPE* Imodel,
                                 int z,
                                 int x)
{
  int iz = blockIdx.z;
  int ix = threadIdx.x + blockIdx.x * blockDim.x;

  if (iz >= z || ix >= x)
    return;

  auto tmp = fic[iz] / den[iz];
  fic[iz] = tmp;
  Imodel[iz * x + ix] = tmp * Imodel[iz * x + ix];
}