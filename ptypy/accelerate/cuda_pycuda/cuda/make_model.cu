#include <thrust/complex.h>
using thrust::complex;

extern "C" __global__ void make_model(
    const CTYPE* in, FTYPE* out, int z, int y, int x)
{
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iz = blockIdx.z;

  if (ix >= x)
    return;

  // we sum accross y directly, as this is the number of modes,
  // which is typically small
  auto sum = FTYPE();
  for (auto iy = 0; iy < y; ++iy)
  {
    auto v = in[iz * y * x + iy * x + ix];
    sum += v.real() * v.real() + v.imag() * v.imag();
  }
  out[iz * x + ix] = sum;
}