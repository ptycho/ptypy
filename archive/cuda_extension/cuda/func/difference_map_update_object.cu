#include "difference_map_update_object.h"
#include "utils/Complex.h"
#include "utils/GpuManager.h"
#include "utils/ScopedTimer.h"

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

/********** Class implementation **********/

DifferenceMapUpdateObject::DifferenceMapUpdateObject()
    : CudaFunction("difference_map_update_object")
{
}

void DifferenceMapUpdateObject::setParameters(int A,
                                              int B,
                                              int C,
                                              int D,
                                              int E,
                                              int F,
                                              int G,
                                              int H,
                                              int I,
                                              float obj_smooth_std,
                                              bool doSmoothing,
                                              bool doClipping)
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
  doSmoothing_ = doSmoothing;
  doClipping_ = doClipping;

  extract_array_from_exit_wave_ =
      gpuManager.get_cuda_function<ExtractArrayFromExitWave>(
          "dm_update_object.extract_array_from_exit_wave",
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

  if (doClipping_)
  {
    clip_complex_magnitudes_to_range_ =
        gpuManager.get_cuda_function<ClipComplexMagnitudesToRange>(
            "dm_update_object.clip_complex_magnitudes_to_range", G_ * H_ * I_);
  }
  if (doSmoothing_)
  {
    int dims[] = {G_, H_, I_};
    float mfs[] = {obj_smooth_std, obj_smooth_std};
    gaussian_filter_ = gpuManager.get_cuda_function<ComplexGaussianFilter>(
        "dm_update_object.gaussian_filter", 3, dims, mfs);
  }
}

void DifferenceMapUpdateObject::setDeviceBuffers(complex<float>* d_obj,
                                                 float* d_object_weights,
                                                 complex<float>* d_probe,
                                                 complex<float>* d_exit_wave,
                                                 int* d_addr_info,
                                                 complex<float>* d_cfact)
{
  d_obj_ = d_obj;
  d_object_weights_ = d_object_weights;
  d_probe_ = d_probe;
  d_exit_wave_ = d_exit_wave;
  d_addr_info_ = d_addr_info;
  d_cfact_ = d_cfact;
}

void DifferenceMapUpdateObject::allocate()
{
  ScopedTimer t(this, "allocate");
  d_obj_.allocate(G_ * H_ * I_);
  d_object_weights_.allocate(G_);
  d_probe_.allocate(D_ * E_ * F_);
  d_exit_wave_.allocate(A_ * B_ * C_);
  d_addr_info_.allocate(A_ * 15);
  d_cfact_.allocate(G_ * H_ * I_);

  extract_array_from_exit_wave_->setDeviceBuffers(d_exit_wave_.get(),
                                                  d_addr_info_.get() + 6,
                                                  d_probe_.get(),
                                                  d_addr_info_.get(),
                                                  d_obj_.get(),
                                                  d_addr_info_.get() + 3,
                                                  d_object_weights_.get(),
                                                  d_cfact_.get(),
                                                  nullptr);

  extract_array_from_exit_wave_->allocate();

  if (doClipping_)
  {
    clip_complex_magnitudes_to_range_->setDeviceBuffers(d_obj_.get());
    clip_complex_magnitudes_to_range_->allocate();
  }

  if (doSmoothing_)
  {
    gaussian_filter_->setDeviceBuffers(d_obj_.get(), d_obj_.get());
    gaussian_filter_->allocate();
  }
}

void DifferenceMapUpdateObject::transfer_in(const complex<float>* obj,
                                            const float* object_weigths,
                                            const complex<float>* probe,
                                            const complex<float>* exit_wave,
                                            const int* addr_info,
                                            const complex<float>* cfact)
{
  ScopedTimer t(this, "transfer in");
  gpu_memcpy_h2d(d_obj_.get(), obj, G_ * H_ * I_);
  gpu_memcpy_h2d(d_object_weights_.get(), object_weigths, G_);
  gpu_memcpy_h2d(d_probe_.get(), probe, D_ * E_ * F_);
  gpu_memcpy_h2d(d_exit_wave_.get(), exit_wave, A_ * B_ * C_);
  gpu_memcpy_h2d(d_addr_info_.get(), addr_info, A_ * 15);
  gpu_memcpy_h2d(d_cfact_.get(), cfact, G_ * H_ * I_);
}

void DifferenceMapUpdateObject::run(float clip_min, float clip_max)
{
  ScopedTimer t(this, "run");

  if (doSmoothing_)
  {
    gaussian_filter_->run();
  }

  int total = G_ * H_ * I_;
  int threadsPerBlock = 256;
  int blocks = (total + threadsPerBlock - 1) / threadsPerBlock;
  multiply_kernel<<<blocks, threadsPerBlock>>>(
      d_obj_.get(), d_cfact_.get(), d_obj_.get(), G_ * H_ * I_);
  checkLaunchErrors();

  extract_array_from_exit_wave_->run();

  if (doClipping_)
  {
    clip_complex_magnitudes_to_range_->run(clip_min, clip_max);
  }

  timing_sync();
}

void DifferenceMapUpdateObject::transfer_out(complex<float>* obj)
{
  ScopedTimer t(this, "transfer out");
  gpu_memcpy_d2h(obj, d_obj_.get(), G_ * H_ * I_);
}

/********** Interface functions ******/

extern "C" void difference_map_update_object_c(
    float* f_obj,                 // G x H x I
    const float* object_weights,  // G
    const float* f_probe,         // D x E x F
    const float* f_exit_wave,     // A x B x C
    const int* addr_info,         // A x 5 x 3
    const float* f_cfact_object,  // G x H x I
    float ob_smooth_std,          // scalar
    float clip_min,               // scalar
    float clip_max,               // scalar
    int doSmoothing,              // boolean if smoothing should be done
    int doClipping,               // boolean if clipping should be done
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
  auto obj = reinterpret_cast<complex<float>*>(f_obj);
  auto probe = reinterpret_cast<const complex<float>*>(f_probe);
  auto exit_wave = reinterpret_cast<const complex<float>*>(f_exit_wave);
  auto cfact = reinterpret_cast<const complex<float>*>(f_cfact_object);

  auto dmuo = gpuManager.get_cuda_function<DifferenceMapUpdateObject>(
      "dm_update_object",
      A,
      B,
      C,
      D,
      E,
      F,
      G,
      H,
      I,
      ob_smooth_std,
      doSmoothing != 0,
      doClipping != 0);

  dmuo->allocate();
  dmuo->transfer_in(obj, object_weights, probe, exit_wave, addr_info, cfact);
  dmuo->run(clip_min, clip_max);
  dmuo->transfer_out(obj);
}