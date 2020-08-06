
extern "C" __global__ void error_reduce(const float* ferr,
                                        float* err_fmag,
                                        int M,
                                        int N)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int batch = blockIdx.x;
  extern __shared__ double sum_v[1024];

  int shidx =
      ty * blockDim.x + tx;  // shidx: index in shared memory for this block
  double sum = 0.0f;

  for (int m = ty; m < M; m += blockDim.y)
  {
#pragma unroll(4)
    for (int n = tx; n < N; n += blockDim.x)
    {
      int idx = batch * M * N + m * N +
                n;  // idx is index qwith respect to the full stack
      sum += ferr[idx];
    }
  }

  sum_v[shidx] = sum;

  __syncthreads();

  int nt = blockDim.x * blockDim.y;
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
    err_fmag[batch] = float(sum_v[0]);
  }
}
