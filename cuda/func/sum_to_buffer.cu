#include "sum_to_buffer.h"

#include "addr_info_helpers.h"
#include "utils/Complex.h"
#include "utils/GpuManager.h"
#include "utils/ScopedTimer.h"

#include <iostream>

/********** Kernels ***********************/

template <class T, int BlockX, int BlockY>
__global__ void sum_to_buffer_kernel(T *out,
                                     int os_1,
                                     int os_2,
                                     const T *in1,
                                     int in1_1,
                                     int in1_2,
                                     const int *__restrict__ in1_addr,
                                     const int *outidx,
                                     const int *startidx,
                                     const int *__restrict__ indices,
                                     int addr_stride)
{
  auto outi = outidx[blockIdx.x];
  auto ind_start = indices + startidx[blockIdx.x];
  auto ind_end = indices + startidx[blockIdx.x + 1];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

#pragma unroll(2)
  for (int i = tx; i < in1_1; i += BlockX)
  {
#pragma unroll(1)
    for (int j = ty; j < in1_2; j += BlockY)
    {
      auto oi = outi * os_1 * os_2 + i * os_2 + j;
      T val = T();
      for (auto is = ind_start; is != ind_end; ++is)
      {
        auto i1 = in1_addr + *is * addr_stride;
        auto i1_0 = i1[0];
        auto i1_1 = i1[1] + i;
        auto i1_2 = i1[2] + j;
        val += in1[i1_0 * in1_1 * in1_2 + i1_1 * in1_2 + i1_2];
      }
      out[oi] = val;
    }
  }
}

/********** Class implementation **************/

template <class T>
SumToBuffer<T>::SumToBuffer() : CudaFunction("sum_to_buffer")
{
}

template <class T>
void SumToBuffer<T>::setParameters(int in1_0,
                                   int in1_1,
                                   int in1_2,
                                   int os_0,
                                   int os_1,
                                   int os_2,
                                   int in1_addr_0,
                                   int out1_addr_0)
{
  in1_0_ = in1_0;
  in1_1_ = in1_1;
  in1_2_ = in1_2;
  os_0_ = os_0;
  os_1_ = os_1;
  os_2_ = os_2;
  in1_addr_0_ = in1_addr_0;
  out1_addr_0_ = out1_addr_0;
}

template <class T>
void SumToBuffer<T>::setAddrStride(int stride)
{
  addr_stride_ = stride;
}

template <class T>
void SumToBuffer<T>::setDeviceBuffers(T *d_in1,
                                      T *d_out,
                                      int *d_in1_addr,
                                      int *d_out1_addr,
                                      int *d_outidx,
                                      int *d_startidx,
                                      int *d_indices,
                                      int outidx_size)
{
  d_in1_ = d_in1;
  d_out_ = d_out;
  d_in1_addr_ = d_in1_addr;
  d_out1_addr_ = d_out1_addr;
  d_outidx_ = d_outidx;
  d_startidx_ = d_startidx;
  d_indices_ = d_indices;
  if (d_outidx)
  {
    outidx_size_ = outidx_size;
  }
}

template <class T>
void SumToBuffer<T>::allocate()
{
  ScopedTimer t(this, "allocate");
  d_in1_.allocate(in1_0_ * in1_1_ * in1_2_);
  d_out_.allocate(os_0_ * os_1_ * os_2_);
  d_in1_addr_.allocate(in1_addr_0_ * addr_stride_);
  d_out1_addr_.allocate(out1_addr_0_ * addr_stride_);
  if (!outidx_.empty())
  {
    d_outidx_.allocate(outidx_.size());
    d_startidx_.allocate(startidx_.size());
    d_indices_.allocate(indices_.size());
    outidx_size_ = outidx_.size();
  }
}

template <class T>
int SumToBuffer<T>::calculateAddrIndices(const int *out1_addr)
{
  // calculate the indexing map
  outidx_.clear();
  startidx_.clear();
  indices_.clear();
  flatten_out_addr(
      out1_addr, out1_addr_0_, addr_stride_, outidx_, startidx_, indices_);
  outidx_size_ = outidx_.size();
  return outidx_size_;
}

template <class T>
T *SumToBuffer<T>::getOutput() const
{
  return d_out_.get();
}

template <class T>
void SumToBuffer<T>::transfer_in(const T *in1,
                                 const int *in1_addr,
                                 const int *out1_addr)
{
  ScopedTimer t(this, "transfer in");

  gpu_memcpy_h2d(d_in1_.get(), in1, in1_0_ * in1_1_ * in1_2_);
  gpu_memcpy_h2d(d_in1_addr_.get(), in1_addr, in1_addr_0_ * addr_stride_);
  gpu_memcpy_h2d(d_out1_addr_.get(), out1_addr, out1_addr_0_ * addr_stride_);
  if (!outidx_.empty())
  {
    gpu_memcpy_h2d(d_outidx_.get(), outidx_.data(), outidx_.size());
    gpu_memcpy_h2d(d_startidx_.get(), startidx_.data(), startidx_.size());
    gpu_memcpy_h2d(d_indices_.get(), indices_.data(), indices_.size());
  }
}

template <class T>
void SumToBuffer<T>::run()
{
  ScopedTimer t(this, "run");
  dim3 threadsPerBlock = {32u, 32u, 1u};
  dim3 blocks = {unsigned(outidx_size_), 1u, 1u};
  sum_to_buffer_kernel<T, 32, 32>
      <<<blocks, threadsPerBlock>>>(d_out_.get(),
                                    os_1_,
                                    os_2_,
                                    d_in1_.get(),
                                    in1_1_,
                                    in1_2_,
                                    d_in1_addr_.get(),
                                    d_outidx_.get(),
                                    d_startidx_.get(),
                                    d_indices_.get(),
                                    addr_stride_);
  checkLaunchErrors();

  // sync device if timing is enabled
  timing_sync();
}

template <class T>
void SumToBuffer<T>::transfer_out(T *out)
{
  ScopedTimer t(this, "transfer out");
  gpu_memcpy_d2h(out, d_out_.get(), os_0_ * os_1_ * os_2_);
}

/***** interface function **************/

// instantiate here to force creating all symbols
template class SumToBuffer<float>;
template class SumToBuffer<complex<float>>;

template <class T>
void sum_to_buffer_tc(const T *in1,
                      int in1_0,
                      int in1_1,
                      int in1_2,
                      T *out,
                      int out_0,
                      int out_1,
                      int out_2,
                      const int *in_addr,
                      int in_addr_0,
                      const int *out_addr,
                      int out_addr_0)
{
  auto s2b = gpuManager.get_cuda_function<SumToBuffer<T>>(
      "sum_to_buffer<" + getTypeName<T>() + ">",
      in1_0,
      in1_1,
      in1_2,
      out_0,
      out_1,
      out_2,
      in_addr_0,
      out_addr_0);
  s2b->calculateAddrIndices(out_addr);
  s2b->allocate();
  s2b->transfer_in(in1, in_addr, out_addr);
  s2b->run();
  s2b->transfer_out(out);
}

extern "C" void sum_to_buffer_c(const float *in1,
                                int in1_0,
                                int in1_1,
                                int in1_2,
                                float *out,
                                int out_0,
                                int out_1,
                                int out_2,
                                const int *in_addr,
                                int in_addr_0,
                                const int *out_addr,
                                int out_addr_0,
                                int isComplex)
{
  if (isComplex != 0)
  {
    sum_to_buffer_tc(reinterpret_cast<const complex<float> *>(in1),
                     in1_0,
                     in1_1,
                     in1_2,
                     reinterpret_cast<complex<float> *>(out),
                     out_0,
                     out_1,
                     out_2,
                     in_addr,
                     in_addr_0,
                     out_addr,
                     out_addr_0);
  }
  else
  {
    sum_to_buffer_tc(in1,
                     in1_0,
                     in1_1,
                     in1_2,
                     out,
                     out_0,
                     out_1,
                     out_2,
                     in_addr,
                     in_addr_0,
                     out_addr,
                     out_addr_0);
  }
}

template <class T>
void sum_to_buffer_stride_tc(const T *in1,
                             int in1_0,
                             int in1_1,
                             int in1_2,
                             T *out,
                             int out_0,
                             int out_1,
                             int out_2,
                             const int *addr_info,
                             int addr_info_0)
{
  auto s2b = gpuManager.get_cuda_function<SumToBuffer<T>>(
      "sum_to_buffer<" + getTypeName<T>() + ">",
      in1_0,
      in1_1,
      in1_2,
      out_0,
      out_1,
      out_2,
      addr_info_0,
      addr_info_0);
  s2b->setAddrStride(15);
  auto in_addr = addr_info + 6;
  auto out_addr = addr_info + 9;
  s2b->calculateAddrIndices(out_addr);
  s2b->allocate();
  s2b->transfer_in(in1, in_addr, out_addr);
  s2b->run();
  s2b->transfer_out(out);
}

extern "C" void sum_to_buffer_stride_c(const float *in1,
                                       int in1_0,
                                       int in1_1,
                                       int in1_2,
                                       float *out,
                                       int out_0,
                                       int out_1,
                                       int out_2,
                                       const int *addr_info,
                                       int addr_info_0,
                                       int isComplex)
{
  if (isComplex != 0)
  {
    sum_to_buffer_stride_tc(reinterpret_cast<const complex<float> *>(in1),
                            in1_0,
                            in1_1,
                            in1_2,
                            reinterpret_cast<complex<float> *>(out),
                            out_0,
                            out_1,
                            out_2,
                            addr_info,
                            addr_info_0);
  }
  else
  {
    sum_to_buffer_stride_tc(in1,
                            in1_0,
                            in1_1,
                            in1_2,
                            out,
                            out_0,
                            out_1,
                            out_2,
                            addr_info,
                            addr_info_0);
  }
}