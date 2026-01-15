/** intens_renorm - with 2 steps as separate kernels.
 *
 * Data types:
 * - IN_TYPE: the data type for the inputs (float or double)
 * - OUT_TYPE: the data type for the outputs (float or double)
 * - MATH_TYPE: the data type used for computation 
 */

#include "common.cuh"

extern "C" __global__ void step1(const IN_TYPE* Imodel,
                                 const IN_TYPE* I,
                                 const IN_TYPE* w,
                                 OUT_TYPE* num,
                                 OUT_TYPE* den,
                                 int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= n)
    return;

  auto tmp = MATH_TYPE(w[i]) * MATH_TYPE(Imodel[i]);
  num[i] = tmp * MATH_TYPE(I[i]);
  den[i] = tmp * MATH_TYPE(Imodel[i]);
}

extern "C" __global__ void step2(const IN_TYPE* fic_tmp,
                                 OUT_TYPE* fic,
                                 OUT_TYPE* Imodel,
                                 int X,
                                 int Y)
{
  int iz = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  // one thread block per fic data point - we want the first thread to read this
  // into shared memory and then sync the block, so we don't get into data races
  // with writing it back to global memory in the end (and we read the value only
  // once)
  //
  __shared__ MATH_TYPE shfic[1];
  if (tx == 0 && ty == 0) {
    shfic[0] = MATH_TYPE(fic[iz]) / MATH_TYPE(fic_tmp[iz]);
  } 
  __syncthreads();

  // now all threads can access that value
  auto tmp = shfic[0];

  // offset Imodel for current z
  Imodel += iz * X * Y;
  
  for (int iy = ty; iy < Y; iy += blockDim.y) {
    #pragma unroll(4)
    for (int ix = tx; ix < X; ix += blockDim.x) {
      Imodel[iy * X + ix] *= tmp;
    }
  }
    
  // race condition if write is not restricted to one thread
  if (tx==0 && ty == 0)
    fic[iz] = tmp;
}
