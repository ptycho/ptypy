/** fmag_all_update.
 *
 * Data types:
 * - IN_TYPE: the data type for the inputs (float or double)
 * - OUT_TYPE: the data type for the outputs (float or double)
 * - MATH_TYPE: the data type used for computation 
 * - ACC_TYPE: data type used for accumulation
 */

#include "common.cuh"

extern "C" __global__ void make_a012(const complex<IN_TYPE>* f,
                                     const complex<IN_TYPE>* a,
                                     const complex<IN_TYPE>* b,
                                     const IN_TYPE* I,
                                     const IN_TYPE* fic,
                                     OUT_TYPE* A0,
                                     OUT_TYPE* A1,
                                     OUT_TYPE* A2,
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
    A0[iz * x + ix] = OUT_TYPE(0);  // make sure it's the right type (double/float)
    A1[iz * x + ix] = OUT_TYPE(0);
    A2[iz * x + ix] = OUT_TYPE(0);
    return;
  }

  // we sum across y directly, as this is the number of modes,
  // which is typically small
  auto sumtf0 = ACC_TYPE(0);
  auto sumtf1 = ACC_TYPE(0);
  auto sumtf2 = ACC_TYPE(0);
  for (auto iy = 0; iy < y; ++iy)
  {
    complex<MATH_TYPE> fv = f[iz * y * x + iy * x + ix];
    sumtf0 += fv.real() * fv.real() + fv.imag() * fv.imag();

    complex<MATH_TYPE> av = a[iz * y * x + iy * x + ix];
    // 2 * real(f * conj(a))
    sumtf1 += MATH_TYPE(2) * (fv.real() * av.real() + fv.imag() * av.imag());

    // use FTYPE(2) to make sure double creeps into a float calculation
    // as 2.0 * would make everything double.
    complex<MATH_TYPE> bv = b[iz * y * x + iy * x + ix];
    // 2 * real(f * conj(b)) + abs(a)^2
    sumtf2 += MATH_TYPE(2) * (fv.real() * bv.real() + fv.imag() * bv.imag()) +
              (av.real() * av.real() + av.imag() * av.imag());
  }

  MATH_TYPE Iv = I[iz * x + ix];
  MATH_TYPE ficv = fic[iz];
  A0[iz * x + ix] = OUT_TYPE(MATH_TYPE(sumtf0) * ficv - Iv);
  A1[iz * x + ix] = OUT_TYPE(MATH_TYPE(sumtf1) * ficv);
  A2[iz * x + ix] = OUT_TYPE(MATH_TYPE(sumtf2) * ficv);
}