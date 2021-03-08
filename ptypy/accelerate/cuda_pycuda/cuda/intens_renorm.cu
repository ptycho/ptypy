/** intens_renorm - with 2 steps as separate kernels.
 *
 * Data types:
 * - IN_TYPE: the data type for the inputs (float or double)
 * - OUT_TYPE: the data type for the outputs (float or double)
 * - MATH_TYPE: the data type used for computation 
 */

#include <thrust/complex.h>
using thrust::complex;

extern "C" __global__ void step1(const IN_TYPE* Imodel,
                                   const IN_TYPE* I,
                                   const IN_TYPE* w,
                                   OUT_TYPE* num,
                                   OUT_TYPE* den,
                                   int z,
                                   int x)
{
  int iz = blockIdx.z;
  int ix = threadIdx.x + blockIdx.x * blockDim.x;

  if (iz >= z || ix >= x)
    return;

  auto tmp = MATH_TYPE(w[iz * x + ix]) * MATH_TYPE(Imodel[iz * x + ix]);
  num[iz * x + ix] = tmp * MATH_TYPE(I[iz * x + ix]);
  den[iz * x + ix] = tmp * MATH_TYPE(Imodel[iz * x + ix]);
}

extern "C" __global__ void step2(const IN_TYPE* fic_nom,
                                 const IN_TYPE* fic_den,
                                 OUT_TYPE* fic,
                                 OUT_TYPE* Imodel,
                                 int z,
                                 int x)
{
  int iz = blockIdx.z;
  int ix = threadIdx.x + blockIdx.x * blockDim.x;

  if (iz >= z || ix >= x)
    return;
  //probably not so clever having all threads read from the same locations
  auto tmp = MATH_TYPE(fic_nom[iz]) / MATH_TYPE(fic_den[iz]);
  Imodel[iz * x + ix] *= tmp;
  // race condition if write is not restricted to one thread
  // learned this the hard way
  if (ix==0)
    fic[iz] = tmp;
}