/** gd_main kernel.
 *
 * Data types:
 * - IN_TYPE: the data type for the inputs (float or double)
 * - OUT_TYPE: the data type for the outputs (float or double - for aux wave)
 * - MATH_TYPE: the data type used for computation 
 */

#include "common.cuh"

extern "C" __global__ void gd_main(const IN_TYPE* Imodel,
                                   const IN_TYPE* I,
                                   const IN_TYPE* w,
                                   OUT_TYPE* err,
                                   complex<OUT_TYPE>* aux,
                                   int z,
                                   int modes,
                                   int x)
{
  int iz = blockIdx.z;
  int ix = threadIdx.x + blockIdx.x * blockDim.x;

  if (iz >= z || ix >= x)
    return;

  auto DI = MATH_TYPE(Imodel[iz * x + ix]) - MATH_TYPE(I[iz * x + ix]);
  auto tmp = MATH_TYPE(w[iz * x + ix]) * MATH_TYPE(DI);
  err[iz * x + ix] = tmp * DI;

  // now set this for all modes (promote)
  for (int m = 0; m < modes; ++m)
  {
    aux[iz * x * modes + m * x + ix] *= tmp;
  }
}