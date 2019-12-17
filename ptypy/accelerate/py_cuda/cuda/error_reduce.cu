#include <iostream>
#include <utility>
#include <thrust/complex.h>
#include <stdio.h>


extern "C"{
__global__ void error_reduce(float *ferr,
                                  float *err_fmag,
                                  int M,
                                  int N)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int batch = blockIdx.x;
  extern __shared__ float sum_v[];

  int shidx = tx * blockDim.y + ty; // shidx is the index in shared memory for this single block
  sum_v[shidx] = 0.0;

  for (int m = tx; m < M; m += blockDim.x)
  {
    for (int n = ty; n < N; n += blockDim.y)
    {
      int idx = batch * M * N + m * N + n; // idx is index qwith respect to the full stack
      sum_v[shidx] += ferr[idx];
    }
  }


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
}