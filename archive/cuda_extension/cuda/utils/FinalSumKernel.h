
// to be run in a single thread block
template <class T>
__global__ void final_sum(const T* data, T* output, int n)
{
  int tid = threadIdx.x;
  extern __shared__ T sum[];

  auto val = 0.0f;
  // in case we have more data than threads in this block
  for (int i = tid; i < n; i += blockDim.x)
  {
    val += data[i];
  }
  sum[tid] = val;

  // now add up sumbuffer in shared memory
  __syncthreads();
  int nt = blockDim.x;
  int c = nt;
  while (c > 1)
  {
    int half = c / 2;
    if (tid < half)
    {
      sum[tid] += sum[c - tid - 1];
    }
    __syncthreads();
    c = c - half;
  }

  if (tid == 0)
  {
    output[0] = sum[0];
  }
}
