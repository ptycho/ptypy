/** error_reduce kernel.
 *
 * Data types:
 * - IN_TYPE: the data type for the inputs (float or double)
 * - OUT_TYPE: the data type for the outputs (float or double)
 * - ACC_TYPE: the data type used for computation 
 */

extern "C" __global__ void error_reduce(const IN_TYPE* ferr,
                                        OUT_TYPE* err_fmag,
                                        int M,
                                        int N)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int batch = blockIdx.x;
  __shared__ ACC_TYPE sum_v[BDIM_X*BDIM_Y];

  int shidx =
      ty * blockDim.x + tx;  // shidx: index in shared memory for this block
  ACC_TYPE sum = ACC_TYPE(0.0);

  for (int m = ty; m < M; m += blockDim.y)
  {
#pragma unroll(4)
    for (int n = tx; n < N; n += blockDim.x)
    {
      int idx = batch * M * N + m * N +
                n;  // idx is index qwith respect to the full stack
      sum += ACC_TYPE(ferr[idx]);
    }
  }

  sum_v[shidx] = sum;

  __syncthreads();

  int nt = BDIM_X * BDIM_Y;
  int c = nt;

  while (c > 1)
  {
    int half = c / 2;
    if (shidx < half)
    {
      sum_v[shidx] += sum_v[c - shidx - 1];
    }
    __syncthreads();
    c = c - half;
  }

  if (shidx == 0)
  {
    err_fmag[batch] = OUT_TYPE(sum_v[0]);
  }
}
