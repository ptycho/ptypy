/** fmag_all_update.
 *
 * Data types:
 * - IN_TYPE: the data type for the inputs (float or double)
 * - OUT_TYPE: the data type for the outputs (float or double)
 * - MATH_TYPE: the data type used for computation 
 */

#include "common.cuh"

extern "C" __global__ void fmag_all_update(complex<OUT_TYPE>* f,
                                           const IN_TYPE* fmask,
                                           const IN_TYPE* fmag,
                                           const IN_TYPE* fdev,
                                           const IN_TYPE* err_fmag,
                                           const int* addr_info,
                                           IN_TYPE pbound_,
                                           int A,
                                           int B)
{
  int batch = blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int addr_stride = 15;
  MATH_TYPE pbound = pbound_;

  const int* ea = addr_info + batch * addr_stride + 6;
  const int* da = addr_info + batch * addr_stride + 9;
  const int* ma = addr_info + batch * addr_stride + 12;

  fmask += ma[0] * A * B;
  MATH_TYPE err = err_fmag[da[0]];
  fdev += da[0] * A * B;
  fmag += da[0] * A * B;
  f += ea[0] * A * B;
  MATH_TYPE renorm = sqrt(pbound / err);

  for (int a = ty; a < A; a += blockDim.y)
  {
    for (int b = tx; b < B; b += blockDim.x)
    {
      MATH_TYPE m = fmask[a * A + b];
      if (renorm < 1.0f)
      {
        /*
        // assuming this is actually a mask, i.e. 0 or 1 --> this is slower
        float fm = m < 0.5f ? 1.0f :
          ((fmag[a * A + b] + fdev[a * A + b] * renorm) / (fdev[a * A + b] +
        fmag[a * A + b]  + 1e-7f)) ;
        */
        MATH_TYPE fmagv = fmag[a * A + b];
        MATH_TYPE fdevv = fdev[a * A + b];
        MATH_TYPE fm = (MATH_TYPE(1) - m) +
                   m * ((fmagv + fdevv * renorm) / (fmagv + fdevv + MATH_TYPE(1e-7)));
        f[a * A + b] *= fm;
      }
    }
  }
}
