/** full_reduce kernel.
 *
 * Data types:
 * - IN_TYPE: the data type for the inputs (float or double)
 * - OUT_TYPE: the data type for the outputs (float or double - for aux wave)
 * - ACC_TYPE: the data type used for internal accumulation
 */


extern "C" __global__ void full_reduce(const IN_TYPE* in, OUT_TYPE* out, int size)
{
  assert(gridDim.x == 1);
  int tx = threadIdx.x;

  __shared__ ACC_TYPE smem[BDIM_X];

  auto sum = ACC_TYPE();
  for (int ix = tx; ix < size; ix += blockDim.x)
  {
    sum = sum + ACC_TYPE(in[ix]);
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
    out[0] = OUT_TYPE(smem[0]);
  }
}