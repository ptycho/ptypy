#include "addr_info_helpers.h"
#include "extract_array_from_exit_wave.h"
#include "utils/GpuManager.h"
#include "utils/ScopedTimer.h"

#include <cassert>
#include <cstdio>
#include <iostream>

/*********** kernels ************************/

__device__ inline void atomicAdd(complex<float>* x, complex<float> y)
{
  auto xf = reinterpret_cast<float*>(x);
  atomicAdd(xf, y.real());
  atomicAdd(xf + 1, y.imag());
}

template <int BlockX, int BlockY>
__global__ void extract_array_from_exit_wave_kernel(
    const complex<float>* exit_wave,
    int A,
    int B,
    int C,
    const int* exit_addr,
    const complex<float>* array_to_be_extracted,
    int D,
    int E,
    int F,
    const int* extract_addr,
    complex<float>* array_to_be_updated,
    int G,
    int H,
    int I,
    const int* update_addr,
    const float* weights,
    complex<float>* denominator,
    int addr_stride)
{
  // one block per addr instance
  int bid = blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  auto pa = update_addr + bid * addr_stride;
  auto oa = extract_addr + bid * addr_stride;
  auto ea = exit_addr + bid * addr_stride;

  array_to_be_extracted += oa[0] * E * F + oa[1] * F + oa[2];
  array_to_be_updated += pa[0] * H * I + pa[1] * I + pa[2];
  denominator += pa[0] * H * I + pa[1] * I + pa[2];

  assert(pa[0] * H * I + pa[1] * I + pa[2] + (B - 1) * I + C - 1 < G * H * I);

  auto weight = weights[pa[0]];
  exit_wave += ea[0] * B * C;

  for (int b = tx; b < B; b += blockDim.x)
  {
    for (int c = ty; c < C; c += blockDim.y)
    {
      auto extracted_array = array_to_be_extracted[b * F + c];
      auto extracted_array_conj = conj(extracted_array);
      atomicAdd(&array_to_be_updated[b * I + c],
                extracted_array_conj * exit_wave[b * C + c] * weight);
      atomicAdd(&denominator[b * I + c],
                extracted_array * extracted_array_conj * weight);
    }
  }
}

template <int BlockX>
__global__ void div_by_denominator(complex<float>* array_to_be_updated,
                                   const complex<float>* denominator,
                                   int n)
{
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= n)
    return;
  array_to_be_updated[gid] /= denominator[gid];
}

/*********** class implementation *********/

ExtractArrayFromExitWave::ExtractArrayFromExitWave()
    : CudaFunction("extract_array_from_exit_wave")

{
}

void ExtractArrayFromExitWave::setParameters(
    int A, int B, int C, int D, int E, int F, int G, int H, int I)
{
  A_ = A;
  B_ = B;
  C_ = C;
  D_ = D;
  E_ = E;
  F_ = F;
  G_ = G;
  H_ = H;
  I_ = I;
}

void ExtractArrayFromExitWave::setDeviceBuffers(
    complex<float>* d_exit_wave,
    int* d_exit_addr,
    complex<float>* d_array_to_be_extracted,
    int* d_extract_addr,
    complex<float>* d_array_to_be_updated,
    int* d_update_addr,
    float* d_weights,
    complex<float>* d_cfact,
    complex<float>* d_denominator)
{
  d_exit_wave_ = d_exit_wave;
  d_exit_addr_ = d_exit_addr;
  d_array_to_be_extracted_ = d_array_to_be_extracted;
  d_extract_addr_ = d_extract_addr;
  d_array_to_be_updated_ = d_array_to_be_updated;
  d_update_addr_ = d_update_addr;
  d_weights_ = d_weights;
  d_cfact_ = d_cfact;
  d_denominator_ = d_denominator;
}

void ExtractArrayFromExitWave::allocate()
{
  ScopedTimer t(this, "allocate");

  d_exit_wave_.allocate(A_ * B_ * C_);
  d_exit_addr_.allocate(A_ * addr_stride_);
  d_array_to_be_extracted_.allocate(D_ * E_ * F_);
  d_extract_addr_.allocate(A_ * addr_stride_);
  d_array_to_be_updated_.allocate(G_ * H_ * I_);
  d_update_addr_.allocate(A_ * addr_stride_);
  d_weights_.allocate(G_);
  d_cfact_.allocate(G_ * H_ * I_);
  d_denominator_.allocate(G_ * H_ * I_);
}

void ExtractArrayFromExitWave::transfer_in(
    const complex<float>* exit_wave,
    const int* exit_addr,
    const complex<float>* array_to_be_extracted,
    const int* extract_addr,
    const complex<float>* array_to_be_updated,
    const int* update_addr,
    const float* weights,
    const complex<float>* cfact)
{
  ScopedTimer t(this, "transfer in");

  gpu_memcpy_h2d(d_exit_wave_.get(), exit_wave, A_ * B_ * C_);
  gpu_memcpy_h2d(d_exit_addr_.get(), exit_addr, A_ * addr_stride_);
  gpu_memcpy_h2d(
      d_array_to_be_extracted_.get(), array_to_be_extracted, D_ * E_ * F_);
  gpu_memcpy_h2d(d_extract_addr_.get(), extract_addr, A_ * addr_stride_);
  gpu_memcpy_h2d(
      d_array_to_be_updated_.get(), array_to_be_updated, G_ * H_ * I_);
  gpu_memcpy_h2d(d_update_addr_.get(), update_addr, A_ * addr_stride_);
  gpu_memcpy_h2d(d_weights_.get(), weights, G_);
  gpu_memcpy_h2d(d_cfact_.get(), cfact, G_ * H_ * I_);
}

void ExtractArrayFromExitWave::run()
{
  ScopedTimer t(this, "run");

  gpu_memcpy_d2d(d_denominator_.get(), d_cfact_.get(), G_ * H_ * I_);

  // we used one block per updateidx
  dim3 threadsPerBlock = {32u, 32u, 1u};
  dim3 blocks = {unsigned(A_), 1u, 1u};
  extract_array_from_exit_wave_kernel<32, 32>
      <<<blocks, threadsPerBlock>>>(d_exit_wave_.get(),
                                    A_,
                                    B_,
                                    C_,
                                    d_exit_addr_.get(),
                                    d_array_to_be_extracted_.get(),
                                    D_,
                                    E_,
                                    F_,
                                    d_extract_addr_.get(),
                                    d_array_to_be_updated_.get(),
                                    G_,
                                    H_,
                                    I_,
                                    d_update_addr_.get(),
                                    d_weights_.get(),
                                    d_denominator_.get(),
                                    addr_stride_);
  checkLaunchErrors();

  int total = G_ * H_ * I_;
  int blocks2 = (total + 255) / 256;
  div_by_denominator<256><<<blocks2, 256>>>(
      d_array_to_be_updated_.get(), d_denominator_.get(), total);

  checkLaunchErrors();
  timing_sync();
}

void ExtractArrayFromExitWave::transfer_out(complex<float>* array_to_be_updated)
{
  ScopedTimer t(this, "transfer out");

  gpu_memcpy_d2h(
      array_to_be_updated, d_array_to_be_updated_.get(), G_ * H_ * I_);
}

/************ interface *******************/

extern "C" void extract_array_from_exit_wave_c(
    const float* f_exit_wave,  // complex
    int A,
    int B,
    int C,
    const int* exit_addr,                  // A x 3 - int
    const float* f_array_to_be_extracted,  // complex
    int D,
    int E,
    int F,
    const int* extract_addr,       // A x 3 - int
    float* f_array_to_be_updated,  // complex
    int G,
    int H,
    int I,
    const int* update_addr,  // A x 3 - int
    const float* weights,    // G  - real
    const float* f_cfact     // G, H, I - complex
)
{
  auto exit_wave = reinterpret_cast<const complex<float>*>(f_exit_wave);
  auto array_to_be_extracted =
      reinterpret_cast<const complex<float>*>(f_array_to_be_extracted);
  auto array_to_be_updated =
      reinterpret_cast<complex<float>*>(f_array_to_be_updated);
  auto cfact = reinterpret_cast<const complex<float>*>(f_cfact);

  auto ex = gpuManager.get_cuda_function<ExtractArrayFromExitWave>(
      "extract_array_from_exit_wave", A, B, C, D, E, F, G, H, I);
  ex->allocate();
  ex->transfer_in(exit_wave,
                  exit_addr,
                  array_to_be_extracted,
                  extract_addr,
                  array_to_be_updated,
                  update_addr,
                  weights,
                  cfact);
  ex->run();
  ex->transfer_out(array_to_be_updated);
}