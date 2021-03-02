/** fill_b_reduce - for second-stage reduction used after fill_b.
 * 
 * Note that the IN_TYPE here must match what's produced by the fill_b kernel
 * Data types:
 * - IN_TYPE: the data type for the inputs
 * - OUT_TYPE: the data type for the outputs
 * - ACC_TYPE: the accumulator type for summing
 */

#include <cassert>

extern "C" __global__ void fill_b_reduce(const IN_TYPE* in, OUT_TYPE* B, int blocks)
{
  // always a single thread block for 2nd stage
  assert(gridDim.x == 1);
  int tx = threadIdx.x;

  __shared__ ACC_TYPE smem[3][BDIM_X];

  double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0;
  for (int ix = tx; ix < blocks; ix += blockDim.x)
  {
    sum0 += in[ix * 3 + 0];
    sum1 += in[ix * 3 + 1];
    sum2 += in[ix * 3 + 2];
  }
  smem[0][tx] = sum0;
  smem[1][tx] = sum1;
  smem[2][tx] = sum2;
  __syncthreads();

  int nt = blockDim.x;
  int c = nt;
  while (c > 1)
  {
    int half = c / 2;
    if (tx < half)
    {
      smem[0][tx] += smem[0][c - tx - 1];
      smem[1][tx] += smem[1][c - tx - 1];
      smem[2][tx] += smem[2][c - tx - 1];
    }
    __syncthreads();
    c = c - half;
  }

  if (tx == 0)
  {
    B[0] += OUT_TYPE(smem[0][0]);
    B[1] += OUT_TYPE(smem[1][0]);
    B[2] += OUT_TYPE(smem[2][0]);
  }
}
