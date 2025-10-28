#include "scan_and_multiply.h"

#include "utils/GpuManager.h"
#include "utils/ScopedTimer.h"

#include <cassert>

/*********** Kernels ******************/

template <int BlockX, int BlockY>
__global__ void scan_and_multiply_kernel(
    complex<float> *out,
    const int *addr_info,  // note: __restrict__ (texture cache) makes it slower
    const complex<float> *probe,
    const complex<float> *obj,
    int probe_m,
    int probe_n,
    int obj_m,
    int obj_n,
    int m,
    int n)
{
  int batch = blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // each of these are 3-d arrays with indices
  auto pa = addr_info + batch * 3 * 5;
  auto oa = pa + 3;
  auto ea = pa + 6;

  // these are the start indices for the batch item - m x n array from here
  // probe start
  auto p_i0 = pa[0];
  auto p_i1 = pa[1];
  auto p_i2 = pa[2];

  // obj start
  auto o_i0 = oa[0];
  auto o_i1 = oa[1];
  auto o_i2 = oa[2];

  // output start
  auto po_i0 = ea[0];

  auto ooffset = o_i0 * obj_m * obj_n + o_i1 * obj_n + o_i2;
  auto poffset = p_i0 * probe_m * probe_n + p_i1 * probe_n + p_i2;
  auto pooffset = po_i0 * m * n;

// let each thread jump by blockDim.x / y, so a 32x32 thread block works on
// any shape of m x n data
#pragma unroll(2)
  for (int i = tx; i < m; i += BlockX)
  {
#pragma unroll(1)
    for (int j = ty; j < n; j += BlockY)
    {
      auto oidx = ooffset + i * obj_n + j;
      auto pidx = poffset + i * probe_n + j;
      auto poidx = pooffset + i * n + j;

      out[poidx] = probe[pidx] * obj[oidx];
    }
  }
}

/*********** Class implementation *****************/

ScanAndMultiply::ScanAndMultiply() : CudaFunction("scan_and_multiply") {}

void ScanAndMultiply::setParameters(int batch_size,
                                    int m,
                                    int n,
                                    int probe_i,
                                    int probe_m,
                                    int probe_n,
                                    int obj_i,
                                    int obj_m,
                                    int obj_n,
                                    int addr_len)
{
  batch_size_ = batch_size;
  m_ = m;
  n_ = n;
  probe_i_ = probe_i;
  probe_m_ = probe_m;
  probe_n_ = probe_n;
  obj_i_ = obj_i;
  obj_m_ = obj_m;
  obj_n_ = obj_n;
  addr_len_ = addr_len;
}

void ScanAndMultiply::setDeviceBuffers(complex<float> *d_probe,
                                       complex<float> *d_obj,
                                       int *d_addr_info,
                                       complex<float> *d_out)
{
  d_probe_ = d_probe;
  d_obj_ = d_obj;
  d_addr_info_ = d_addr_info;
  d_out_ = d_out;
}

void ScanAndMultiply::allocate()
{
  ScopedTimer t(this, "allocate");
  d_out_.allocate(batch_size_ * m_ * n_);
  d_addr_info_.allocate(addr_len_ * 5 * 3);
  d_probe_.allocate(probe_i_ * probe_m_ * probe_n_);
  d_obj_.allocate(obj_i_ * obj_m_ * obj_n_);
}

complex<float> *ScanAndMultiply::getOutput() const { return d_out_.get(); }

void ScanAndMultiply::transfer_in(const complex<float> *probe,
                                  const complex<float> *obj,
                                  const int *addr_info)
{
  ScopedTimer t(this, "transfer in");
  gpu_memcpy_h2d(d_probe_.get(), probe, probe_i_ * probe_m_ * probe_n_);
  gpu_memcpy_h2d(d_obj_.get(), obj, obj_i_ * obj_m_ * obj_n_);
  gpu_memcpy_h2d(d_addr_info_.get(), addr_info, addr_len_ * 5 * 3);
}

void ScanAndMultiply::transfer_out(complex<float> *out)
{
  ScopedTimer t(this, "transfer out");
  gpu_memcpy_d2h(out, d_out_.get(), batch_size_ * m_ * n_);
}

void ScanAndMultiply::run()
{
  ScopedTimer t(this, "run");
  checkCudaErrors(cudaMemset(
      d_out_.get(), 0, batch_size_ * m_ * n_ * sizeof(complex<float>)));
  // always use a 32x32 block of threads
  dim3 threadsPerBlock = {32, 32, 1u};
  dim3 blocks = {unsigned(addr_len_), 1u, 1u};

  scan_and_multiply_kernel<32, 32>
      <<<blocks, threadsPerBlock>>>(d_out_.get(),
                                    d_addr_info_.get(),
                                    d_probe_.get(),
                                    d_obj_.get(),
                                    probe_m_,
                                    probe_n_,
                                    obj_m_,
                                    obj_n_,
                                    m_,
                                    n_);
  checkLaunchErrors();

  // sync device if timing is enabled
  timing_sync();
}

/******* Interface function **********/

extern "C" void scan_and_multiply_c(const float *fprobe,
                                    int probe_i,
                                    int probe_m,
                                    int probe_n,
                                    const float *fobj,
                                    int obj_i,
                                    int obj_m,
                                    int obj_n,
                                    const int *addr_info,
                                    int addr_len,
                                    int batch_size,
                                    int m,
                                    int n,
                                    float *fout)
{
  auto probe = reinterpret_cast<const complex<float> *>(fprobe);
  auto obj = reinterpret_cast<const complex<float> *>(fobj);
  auto out = reinterpret_cast<complex<float> *>(fout);

  auto sam = gpuManager.get_cuda_function<ScanAndMultiply>("scan_and_multiply",
                                                           batch_size,
                                                           m,
                                                           n,
                                                           probe_i,
                                                           probe_m,
                                                           probe_n,
                                                           obj_i,
                                                           obj_m,
                                                           obj_n,
                                                           addr_len);
  sam->allocate();
  sam->transfer_in(probe, obj, addr_info);
  sam->run();
  sam->transfer_out(out);
}
