extern "C" __global__ void indexed_sum_middim(
    const IN_TYPE* data,
    IN_TYPE* sums,
    int i,
    int m,  // dim we work on - 1 block per output
    int n,
    float scale)
{
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  data += bid * n;

  auto val = 0.0f;
  for (int x = 0; x < i; ++x)
  {
    auto d_inner = data + x * n * m;
    for (int z = tid; z < n; z += BDIM_X)
    {
      val += d_inner[z];
    }
  }

  __shared__ float sumshr[BDIM_X];
  sumshr[tid] = val;

  __syncthreads();
  int c = BDIM_X;
  while (c > 1)
  {
    int half = c / 2;
    if (tid < half)
    {
      sumshr[tid] += sumshr[c - tid - 1];
    }
    __syncthreads();
    c = c - half;
  }

  if (tid == 0)
  {
    sums[bid] = sumshr[0] * float(bid) * scale;
  }
}


extern "C" __global__ void indexed_sum_lastdim(
    const IN_TYPE* data, IN_TYPE* sums, int n, int i, float scale)
{
  int ty = threadIdx.y + blockIdx.y * BDIM_Y;
  int tx = threadIdx.x;

  auto val = 0.0f;
  if (ty < i)
  {
    data += ty;  // column to work on

    // we collaborate along the x axis (columns) to get more threads in case i
    // is small
    for (int r = tx; r < n; r += BDIM_X)
    {
      val += data[r * i];
    }
  }

  // reduce along X dimension in shared memory (column sum)
  __shared__ float blocksums[BDIM_X][BDIM_Y];
  blocksums[tx][threadIdx.y] = val;

  __syncthreads();
  int nt = blockDim.x;
  int c = nt;
  while (c > 1)
  {
    int half = c / 2;
    if (tx < half)
    {
      blocksums[tx][threadIdx.y] += blocksums[c - tx - 1][threadIdx.y];
    }
    __syncthreads();
    c = c - half;
  }

  if (ty >= i)
  {
    return;
  }

  if (tx == 0)
  {
    sums[ty] = blocksums[0][threadIdx.y] * float(ty) * scale;
  }
}


extern "C" __global__ void final_sums(const IN_TYPE* sum_i,
                           int i,
                           const IN_TYPE* sum_m,
                           int m,
                           const IN_TYPE* sum_n,
                           int n,
                           IN_TYPE* output)
{
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  // each block works on a single dimension
  int nn = bid == 0 ? i : (bid == 1 ? m : n);
  const float* data = bid == 0 ? sum_i : (bid == 1 ? sum_m : sum_n);

  __shared__ float shared[BDIM_X];
  auto val = 0.0f;
  for (int i = tid; i < nn; i += blockDim.x)
  {
    val += data[i];
  }
  shared[tid] = val;

  // now add up sumbuffer in shared memory
  __syncthreads();
  int nt = blockDim.x;
  int c = nt;
  while (c > 1)
  {
    int half = c / 2;
    if (tid < half)
    {
      shared[tid] += shared[c - tid - 1];
    }
    __syncthreads();
    c = c - half;
  }

  if (tid == 0)
  {
    output[bid] = shared[0];
  }
}

