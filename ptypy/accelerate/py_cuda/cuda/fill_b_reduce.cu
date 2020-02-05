#include <cassert>

extern "C" __global__ void fill_b_reduce(const FTYPE* in, FTYPE* B, int blocks)
{
  // always a single thread block for 2nd stage
  assert(gridDim.x == 1);
  int tx = threadIdx.x;

  __shared__ FTYPE smem[3][BDIM_X];

  auto sum0 = FTYPE(), sum1 = FTYPE(), sum2 = FTYPE();
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
    B[0] += smem[0][0];
    B[1] += smem[1][0];
    B[2] += smem[2][0];
  }
}
