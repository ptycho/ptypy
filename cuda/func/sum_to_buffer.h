#pragma once

#include "utils/CudaFunction.h"
#include "utils/Memory.h"

template <class T>
class SumToBuffer : public CudaFunction
{
public:
  SumToBuffer();
  void setParameters(int in1_0,
                     int in1_1,
                     int in1_2,
                     int os_0,
                     int os_1,
                     int os_2,
                     int in1_addr_0,
                     int out1_addr_0);
  void setAddrStride(int stride);
  int calculateAddrIndices(const int *out1_addr);
  void setDeviceBuffers(T *d_in1,
                        T *d_out,
                        int *d_in1_addr,
                        int *d_out1_addr,
                        int *d_outidx,
                        int *d_startidx,
                        int *d_indices,
                        int outidx_size);
  void allocate();
  T *getOutput() const;
  void transfer_in(const T *in1, const int *in1_addr, const int *out1_addr);
  void run();
  void transfer_out(T *out);

private:
  DevicePtrWrapper<T> d_in1_;
  DevicePtrWrapper<T> d_out_;
  DevicePtrWrapper<int> d_in1_addr_;
  DevicePtrWrapper<int> d_out1_addr_;
  DevicePtrWrapper<int> d_outidx_, d_startidx_, d_indices_;
  std::vector<int> outidx_, startidx_, indices_;
  int in1_0_ = 0, in1_1_ = 0, in1_2_ = 0;
  int os_0_ = 0, os_1_ = 0, os_2_ = 0;
  int in1_addr_0_ = 0, out1_addr_0_ = 0;
  int addr_stride_ = 3;
  int outidx_size_ = 0;
};
