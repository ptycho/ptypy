#include "mass_center.h"
#include "utils/FinalSumKernel.h"
#include "utils/GpuManager.h"
#include "utils/ScopedTimer.h"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

/************ kernels ***************/

template <int BlockX>
__global__ void indexed_sum_middim(
    const float* data,
    float* sums,
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
    for (int z = tid; z < n; z += BlockX)
    {
      val += d_inner[z];
    }
  }

  __shared__ float sumshr[BlockX];
  sumshr[tid] = val;

  __syncthreads();
  int c = BlockX;
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

template <int BlockX, int BlockY>
__global__ void indexed_sum_lastdim(
    const float* data, float* sums, int n, int i, float scale)
{
  int ty = threadIdx.y + blockIdx.y * BlockY;
  int tx = threadIdx.x;

  auto val = 0.0f;
  if (ty < i)
  {
    data += ty;  // column to work on

    // we collaborate along the x axis (columns) to get more threads in case i
    // is small
    for (int r = tx; r < n; r += BlockX)
    {
      val += data[r * i];
    }
  }

  // reduce along X dimension in shared memory (column sum)
  __shared__ float blocksums[BlockX][BlockY];
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

template <int BlockX>
__global__ void final_sums(const float* sum_i,
                           int i,
                           const float* sum_m,
                           int m,
                           const float* sum_n,
                           int n,
                           float* output)
{
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  // each block works on a single dimension
  int nn = bid == 0 ? i : (bid == 1 ? m : n);
  const float* data = bid == 0 ? sum_i : (bid == 1 ? sum_m : sum_n);

  __shared__ float shared[BlockX];
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

/************ class implementation ********/

MassCenter::MassCenter() : CudaFunction("mass_center") {}

void MassCenter::setParameters(int i, int m, int n)
{
  i_ = i;
  m_ = m;
  n_ = n;
}

void MassCenter::setDeviceBuffers(float* d_data, float* d_out)
{
  d_data_ = d_data;
  d_out_ = d_out;
}

void MassCenter::allocate()
{
  ScopedTimer t(this, "allocate");
  d_data_.allocate(i_ * m_ * n_);
  d_i_sum_.allocate(i_);
  d_m_sum_.allocate(m_);
  d_n_sum_.allocate(n_);
  d_out_.allocate(n_ > 1 ? 3 : 2);
}

void MassCenter::transfer_in(const float* data)
{
  ScopedTimer t(this, "transfer in");
  gpu_memcpy_h2d(d_data_.get(), data, i_ * m_ * n_);
}

void MassCenter::run()
{
  ScopedTimer t(this, "run");

  const int threadsPerBlock = 256;

  // first, calculate the total sum of all entries (using thrust)
  thrust::device_ptr<float> raw(d_data_.get());
  auto total_sum = thrust::reduce(raw, raw + i_ * n_ * m_);
  auto sc = 1.0f / total_sum;

  // sum all dims except the first, multiplying by the index and scaling factor
  indexed_sum_middim<threadsPerBlock><<<i_, threadsPerBlock>>>(
      d_data_.get(), d_i_sum_.get(), 1, i_, n_ * m_, sc);
  checkLaunchErrors();

  if (n_ > 2)
  {
    // 3d case

    // sum all dims, except the middle, multiplying by the index and scaling
    // factor
    indexed_sum_middim<threadsPerBlock><<<m_, threadsPerBlock>>>(
        d_data_.get(), d_m_sum_.get(), i_, n_, m_, sc);
    checkLaunchErrors();

    // sum the all dims except the last, multiplying by the index and scaling
    // factor
    dim3 threads = {32u, 32u, 1u};
    dim3 blk = {1u, unsigned(n_ + 32 - 1) / 32u, 1u};
    indexed_sum_lastdim<32, 32>
        <<<blk, threads>>>(d_data_.get(), d_n_sum_.get(), i_ * m_, n_, sc);
    checkLaunchErrors();
  }
  else
  {
    // 2d case

    // sum the all dims except the last, multiplying by the index and scaling
    // factor
    dim3 threads = {32u, 32u, 1u};
    dim3 blk = {1u, unsigned(m_ + 32 - 1) / 32u, 1u};
    indexed_sum_lastdim<32, 32>
        <<<blk, threads>>>(d_data_.get(), d_m_sum_.get(), i_, m_, sc);
    checkLaunchErrors();
  }

  // summing for final results (TODO:: can we combine these?)
  final_sums<256><<<n_> 1 ? 3 : 2, 256>>> (d_i_sum_.get(),
                                           i_,
                                           d_m_sum_.get(),
                                           m_,
                                           d_n_sum_.get(),
                                           n_,
                                           d_out_.get());
  checkLaunchErrors();
}

void MassCenter::transfer_out(float* out)
{
  ScopedTimer t(this, "transfer out");
  gpu_memcpy_d2h(out, d_out_.get(), n_ > 1 ? 3 : 2);
}

/************ interface *******************/

extern "C" void mass_center_c(
    const float* data, int i, int m, int n, float* output)
{
  auto mc = gpuManager.get_cuda_function<MassCenter>("mass_center", i, m, n);
  mc->allocate();
  mc->transfer_in(data);
  mc->run();
  mc->transfer_out(output);
}