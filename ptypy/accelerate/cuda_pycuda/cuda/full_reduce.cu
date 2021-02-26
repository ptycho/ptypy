#include <cassert>

extern "C" __global__ void full_reduce(const DTYPE* in, DTYPE* out, int size)
{
  assert(gridDim.x == 1);
  int tx = threadIdx.x;

  __shared__ DTYPE smem[BDIM_X];

  auto sum = DTYPE();
  for (int ix = tx; ix < size; ix += blockDim.x)
  {
    sum = sum + in[ix];
  }
  smem[tx] = sum;
  __syncthreads();

  int nt = blockDim.x;
  int c = nt;
  while (c > 1)
  {
    int half = c / 2;
    if (tx < half)
    {
      smem[tx] += smem[c - tx - 1];
    }
    __syncthreads();
    c = c - half;
  }

  if (tx == 0)
  {
    out[0] = smem[0];
  }
}