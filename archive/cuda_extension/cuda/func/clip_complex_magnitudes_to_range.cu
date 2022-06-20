#include "clip_complex_magnitudes_to_range.h"
#include "utils/GpuManager.h"
#include "utils/ScopedTimer.h"

#include <cmath>
#include <cstdlib>

/************ Kernels *********************/

__global__ void clip_complex_magnitudes_to_range_kernel(complex<float>* data,
                                                        int n,
                                                        float clip_min,
                                                        float clip_max)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  auto v = data[id];

  auto mag = abs(v);
  auto theta = arg(v);
  if (mag > clip_max)
    mag = clip_max;
  if (mag < clip_min)
    mag = clip_min;
  v = thrust::polar(mag, theta);

  data[id] = v;
}

/************ Class implementation **********/

ClipComplexMagnitudesToRange::ClipComplexMagnitudesToRange()
    : CudaFunction("clip_complex_magnitudes_to_range")
{
}

void ClipComplexMagnitudesToRange::setParameters(int n) { n_ = n; }

void ClipComplexMagnitudesToRange::setDeviceBuffers(complex<float>* d_data)
{
  d_data_ = d_data;
}

void ClipComplexMagnitudesToRange::allocate()
{
  ScopedTimer t(this, "allocate");
  d_data_.allocate(n_);
}

void ClipComplexMagnitudesToRange::transfer_in(const complex<float>* data)
{
  ScopedTimer t(this, "transfer in");
  gpu_memcpy_h2d(d_data_.get(), data, n_);
}

void ClipComplexMagnitudesToRange::run(float clip_min, float clip_max)
{
  ScopedTimer t(this, "run");

  const int threadsPerBlock = 256;
  int blocks = (n_ + threadsPerBlock - 1) / threadsPerBlock;
  clip_complex_magnitudes_to_range_kernel<<<blocks, threadsPerBlock>>>(
      d_data_.get(), n_, clip_min, clip_max);

  checkLaunchErrors();
  timing_sync();
}

void ClipComplexMagnitudesToRange::transfer_out(complex<float>* data)
{
  ScopedTimer t(this, "transfer out");
  gpu_memcpy_d2h(data, d_data_.get(), n_);
}

/************ interface ******************/

extern "C" void clip_complex_magnitudes_to_range_c(float* f_data,
                                                   int n,
                                                   float clip_min,
                                                   float clip_max)
{
  auto data = reinterpret_cast<complex<float>*>(f_data);

  auto ccmr = gpuManager.get_cuda_function<ClipComplexMagnitudesToRange>(
      "clip_complex_magnitudes_to_range", n);
  ccmr->allocate();
  ccmr->transfer_in(data);
  ccmr->run(clip_min, clip_max);
  ccmr->transfer_out(data);
}