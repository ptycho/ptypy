#include "difference_map_update_probe.h"
#include "utils/GpuManager.h"
#include "utils/ScopedTimer.h"

#include <cmath>

/********* kernels *****************/

static __global__ void multiply_kernel(const complex<float>* in1,
                                const complex<float>* in2,
                                complex<float>* out,
                                int n)
{
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= n)
    return;
  out[gid] = in1[gid] * in2[gid];
}

static __global__ void diff_kernel(const complex<float>* a,
                            const complex<float>* b,
                            complex<float>* res,
                            int n)
{
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= n)
    return;
  res[gid] = a[gid] - b[gid];
}

/********* class implementation *********/

DifferenceMapUpdateProbe::DifferenceMapUpdateProbe()
    : CudaFunction("difference_map_update_probe")
{
}

void DifferenceMapUpdateProbe::setParameters(int A,
                                             int B,
                                             int C,
                                             int D,
                                             int E,
                                             int F,
                                             int G,
                                             int H,
                                             int I,
                                             bool withProbeSupport)
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
  withProbeSupport_ = withProbeSupport;

  extract_array_from_exit_wave_ =
      gpuManager.get_cuda_function<ExtractArrayFromExitWave>(
          "dm_update_probe.extract_array_from_exit_wave",
          A_,
          B_,
          C_,
          D_,
          E_,
          F_,
          G_,
          H_,
          I_);
  extract_array_from_exit_wave_->setAddrStride(15);
  norm2_probe_ = gpuManager.get_cuda_function<Norm2<complex<float>>>(
      "dm_update_probe.norm2_probe", G_ * H_ * I_);
  norm2_diff_ = gpuManager.get_cuda_function<Norm2<complex<float>>>(
      "dm_update_probe.norm2_diff", G_ * H_ * I_);
}

void DifferenceMapUpdateProbe::setDeviceBuffers(complex<float>* d_obj,
                                                float* d_probe_weights,
                                                complex<float>* d_probe,
                                                complex<float>* d_exit_wave,
                                                int* d_addr_info,
                                                complex<float>* d_cfact_probe,
                                                complex<float>* d_probe_support)
{
  d_obj_ = d_obj;
  d_probe_weights_ = d_probe_weights;
  d_probe_ = d_probe;
  d_exit_wave_ = d_exit_wave;
  d_addr_info_ = d_addr_info;
  d_cfact_probe_ = d_cfact_probe;
  d_probe_support_ = d_probe_support;
}

void DifferenceMapUpdateProbe::allocate()
{
  ScopedTimer t(this, "allocate");

  if (withProbeSupport_)
  {
    d_probe_support_.allocate(G_ * H_ * I_);
  }

  d_obj_.allocate(D_ * E_ * F_);
  d_probe_weights_.allocate(G_);
  d_probe_.allocate(G_ * H_ * I_);
  d_buffer_.allocate(G_ * H_ * I_);
  d_exit_wave_.allocate(A_ * B_ * C_);
  d_addr_info_.allocate(A_ * 5 * 3);
  d_cfact_probe_.allocate(G_ * H_ * I_);
  
  

  extract_array_from_exit_wave_->setDeviceBuffers(d_exit_wave_.get(),
                                                  d_addr_info_.get() + 6,
                                                  d_obj_.get(),
                                                  d_addr_info_.get() + 3,
                                                  d_buffer_.get(),
                                                  d_addr_info_.get(),
                                                  d_probe_weights_.get(),
                                                  d_cfact_probe_.get(),
                                                  nullptr);
  extract_array_from_exit_wave_->allocate();

  norm2_probe_->setDeviceBuffers(d_buffer_.get(),
                                 nullptr  // just size 1, allocated internally
  );
  norm2_probe_->allocate();

  // here, we'll run d_probe = d_buffer_ - d_probe
  norm2_diff_->setDeviceBuffers(d_probe_.get(),
                                nullptr  // just size 1, allocated internally
  );
  norm2_diff_->allocate();

  // and we'll copy d_buffer_ to d_probe_ in the end
}

void DifferenceMapUpdateProbe::transfer_in(const complex<float>* obj,
                                           const float* probe_weights,
                                           const complex<float>* probe,
                                           const complex<float>* exit_wave,
                                           const int* addr_info,
                                           const complex<float>* cfact_probe,
                                           const complex<float>* probe_support)
{
  ScopedTimer t(this, "transfer in");
  gpu_memcpy_h2d(d_obj_.get(), obj, D_ * E_ * F_);
  gpu_memcpy_h2d(d_probe_weights_.get(), probe_weights, G_);
  gpu_memcpy_h2d(d_probe_.get(), probe, G_ * H_ * I_);
  gpu_memcpy_h2d(d_exit_wave_.get(), exit_wave, A_ * B_ * C_);
  gpu_memcpy_h2d(d_addr_info_.get(), addr_info, A_ * 3 * 5);
  gpu_memcpy_h2d(d_cfact_probe_.get(), cfact_probe, G_ * H_ * I_);
  if (withProbeSupport_)
  {
    gpu_memcpy_h2d(d_probe_support_.get(), probe_support, G_ * H_ * I_);
  }
}

void DifferenceMapUpdateProbe::run()
{
  ScopedTimer t(this, "run");

  int total = G_ * H_ * I_;
  int threadsPerBlock = 256;
  int blocks = (total + threadsPerBlock - 1) / threadsPerBlock;
  multiply_kernel<<<blocks, threadsPerBlock>>>(
      d_probe_.get(), d_cfact_probe_.get(), d_buffer_.get(), G_ * H_ * I_);
  checkLaunchErrors();

  extract_array_from_exit_wave_->run();

  if (withProbeSupport_)
  {
    multiply_kernel<<<total, threadsPerBlock>>>(
        d_buffer_.get(), d_probe_support_.get(), d_buffer_.get(), G_ * H_ * I_);
    checkLaunchErrors();
  }

  // d_probe = d_buffer_ - d_probe
  diff_kernel<<<total, threadsPerBlock>>>(
      d_buffer_.get(), d_probe_.get(), d_probe_.get(), G_ * H_ * I_);
  checkLaunchErrors();

  norm2_probe_->run();
  norm2_diff_->run();

  gpu_memcpy_d2d(d_probe_.get(), d_buffer_.get(), G_ * H_ * I_);

  timing_sync();
}

void DifferenceMapUpdateProbe::transfer_out(complex<float>* probe,
                                            float* change)
{
  ScopedTimer t(this, "transfer out");
  gpu_memcpy_d2h(probe, d_probe_.get(), G_ * H_ * I_);

  float norm2diff, norm2probe;
  gpu_memcpy_d2h(&norm2diff, norm2_diff_->getOutput(), 1);
  gpu_memcpy_d2h(&norm2probe, norm2_probe_->getOutput(), 1);

  *change = std::sqrt(norm2diff / norm2probe / G_);
}

/********* interface functions *********/

extern "C" float difference_map_update_probe_c(
    const float* f_obj,            // D x E x F
    const float* probe_weights,    // G
    float* f_probe,                // G x H x I
    const float* f_exit_wave,      // A x B x C
    const int* addr_info,          // A x 5 x 3
    const float* f_cfact_probe,    // G x H x I
    const float* f_probe_support,  // G x H x I - can be null
    int A,
    int B,
    int C,
    int D,
    int E,
    int F,
    int G,
    int H,
    int I)
{
  auto obj = reinterpret_cast<const complex<float>*>(f_obj);
  auto probe = reinterpret_cast<complex<float>*>(f_probe);
  auto exit_wave = reinterpret_cast<const complex<float>*>(f_exit_wave);
  auto cfact = reinterpret_cast<const complex<float>*>(f_cfact_probe);
  auto probe_support = reinterpret_cast<const complex<float>*>(f_probe_support);

  auto dmup = gpuManager.get_cuda_function<DifferenceMapUpdateProbe>(
      "dm_update_probe", A, B, C, D, E, F, G, H, I, f_probe_support != nullptr);
  dmup->allocate();
  dmup->transfer_in(
      obj, probe_weights, probe, exit_wave, addr_info, cfact, probe_support);
  dmup->run();
  float change;
  dmup->transfer_out(probe, &change);
  return change;
}