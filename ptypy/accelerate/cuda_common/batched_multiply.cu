/** This kernel was used for FFT pre- and post-scaling,
    to test if cuFFT via python is worthwhile.
    It turned out it wasn't.
 * 
 * Data types:
 * - IN_TYPE: the data type for the inputs
 * - OUT_TYPE: the data type for the outputs
 * - MATH_TYPE: the data type used for computation (filter)
 */

#include "common.cuh"

extern "C" __global__ void batched_multiply(const complex<IN_TYPE>* input,
                                            complex<OUT_TYPE>* output,
                                            const complex<MATH_TYPE>* filter,
                                            float scale,
                                            int nBatches,
                                            int rows,
                                            int columns)
{
  int gx = threadIdx.x + blockIdx.x * blockDim.x;
  int gy = threadIdx.y + blockIdx.y * blockDim.y;
  int gz = threadIdx.z + blockIdx.z * blockDim.z;

  if (gx > rows - 1 || gy > columns - 1 || gz > nBatches)
    return;

  auto val = input[gz * rows * columns + gy * rows + gx];

  if (MPY_DO_FILT)  // set at compile-time
  {
    val *= filter[gy * rows + gx];
  }
  if (MPY_DO_SCALE)  // set at compile-time
    val *= scale;
  output[gz * rows * columns + gy * rows + gx] = val;
}