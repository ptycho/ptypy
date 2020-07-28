#include <thrust/complex.h>
using thrust::complex;

extern "C" __global__ void make_a012(const CTYPE* f,
                                     const CTYPE* a,
                                     const CTYPE* b,
                                     const FTYPE* I,
                                     FTYPE* A0,
                                     FTYPE* A1,
                                     FTYPE* A2,
                                     int z,
                                     int y,
                                     int x,
                                     int maxz)
{
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iz = blockIdx.z;

  if (ix >= x)
    return;

  if (iz >= maxz)
  {
    A0[iz * x + ix] = FTYPE(0);  // make sure it's the right type (double/float)
    A1[iz * x + ix] = FTYPE(0);
    A2[iz * x + ix] = FTYPE(0);
    return;
  }

  // we sum accross y directly, as this is the number of modes,
  // which is typically small
  auto sumtf0 = FTYPE(0);
  auto sumtf1 = FTYPE(0);
  auto sumtf2 = FTYPE(0);
  for (auto iy = 0; iy < y; ++iy)
  {
    auto fv = f[iz * y * x + iy * x + ix];
    sumtf0 += fv.real() * fv.real() + fv.imag() * fv.imag();

    auto av = a[iz * y * x + iy * x + ix];
    // 2 * real(f * conj(a))
    sumtf1 += FTYPE(2) * (fv.real() * av.real() + fv.imag() * av.imag());

    // use FTYPE(2) to make sure double creaps into a float calculation
    // as 2.0 * would make everything double.
    auto bv = b[iz * y * x + iy * x + ix];
    // 2 * real(f * conj(b)) + abs(a)^2
    sumtf2 += FTYPE(2) * (fv.real() * bv.real() + fv.imag() * bv.imag()) +
              (av.real() * av.real() + av.imag() * av.imag());
  }

  auto Iv = I[iz * x + ix];
  A0[iz * x + ix] = sumtf0 - Iv;
  A1[iz * x + ix] = sumtf1;
  A2[iz * x + ix] = sumtf2;
}