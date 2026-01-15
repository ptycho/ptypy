/** make_model - with 2 steps as separate kernels.
 *
 * Data types:
 * - IN_TYPE: the data type for the inputs (float or double)
 * - OUT_TYPE: the data type for the outputs (float or double)
 * - MATH_TYPE: the data type used for computation 
 */

#include "common.cuh"

extern "C" __global__ void make_model(
    const complex<IN_TYPE>* in, OUT_TYPE* out, int z, int y, int x)
{
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iz = blockIdx.z;

  if (ix >= x)
    return;

  // we sum accross y directly, as this is the number of modes,
  // which is typically small
  auto sum = MATH_TYPE();
  for (auto iy = 0; iy < y; ++iy)
  {
    complex<MATH_TYPE> v = in[iz * y * x + iy * x + ix];
    sum += v.real() * v.real() + v.imag() * v.imag();
  }
  out[iz * x + ix] = OUT_TYPE(sum);
}