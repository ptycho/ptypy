/** This kernel was used for FFT pre- and post-scaling,
    to test if cuFFT via python is worthwhile.
    It turned out it wasn't.
*/
#include <thrust/complex.h>
using thrust::complex;

extern "C" __global__ void batched_multiply(const complex<float>* input,
                                            complex<float>* output,
                                            const complex<float>* filter,
                                            float scale,
                                            int nBatches,
                                            int rows,
                                            int columns)
{
  int gx = threadIdx.x + blockIdx.x * blockDim.x;
  int gy = threadIdx.y + blockIdx.y * blockDim.y;
  int gz = threadIdx.z + blockIdx.z * blockDim.z;

  if (gx > columns || gy > rows || gz > nBatches)
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