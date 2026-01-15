#include "farfield_propagator.h"

#include "utils/GpuManager.h"
#include "utils/ScopedTimer.h"

#include <iostream>

/**************** Kernels ***************/

// can't do in-place modification of inputs
__global__ void applyPrefilter(const complex<float> *datain,
                               complex<float> *dataout,
                               const complex<float> *__restrict__ filter,
                               size_t batchsize,
                               size_t size)
{
  size_t offset = threadIdx.x + blockIdx.x * blockDim.x;
  if (offset >= batchsize * size)
    return;
  dataout[offset] = datain[offset] * filter[offset % size];
}

// this can always be done in-place
__global__ void applyPostfilter(complex<float> *data,
                                const complex<float> *__restrict__ filter,
                                float sc,
                                size_t batchsize,
                                size_t size)
{
  size_t offset = threadIdx.x + blockIdx.x * blockDim.x;
  size_t total = batchsize * size;
  if (offset >= total)
    return;
  auto val = data[offset];
  if (filter)
  {
    val = filter[offset % size] * val;
  }
  val *= sc;
  data[offset] = val;
}

/*************** Class implementation ********************/

cufftHandle FFTPlanManager::get_or_create_plan(int i, int m, int n)
{
  key_t key{i, m, n};
  if (plans_.find(key) == plans_.end())
  {
    cufftHandle plan;
    checkCudaErrors(cufftCreate(&plan));
    plans_[key] = plan;

    int dims[] = {m, n};
    size_t workSize;
    checkCudaErrors(cufftMakePlanMany(
        plan, 2, dims, 0, 0, 0, 0, 0, 0, CUFFT_C2C, i, &workSize));
#ifndef NDEBUG
    debug_addMemory((void *)long(plan), workSize);
    std::cout << "Made FFT Plan for " << m << "x" << n << ", batch=" << i
              << std::endl;
    std::cout << "Allocated " << (void *)long(plan)
              << ", total: " << double(debug_getMemory()) << std::endl;
#endif
  }

  return plans_[key];
}

void FFTPlanManager::clearCache()
{
  for (auto &item : plans_)
  {
    cufftDestroy(item.second);
#ifndef NDEBUG
    std::cout << "Freeing for FFT plan " << (void *)long(item.second)
              << std::endl;
    debug_freeMemory((void *)long(item.second));
    std::cout << "Total allocated: " << double(debug_getMemory()) << std::endl;
#endif
  }
  plans_.clear();
}

FFTPlanManager::~FFTPlanManager() { clearCache(); }

/******************************/

FFTPlanManager FarfieldPropagator::planManager_;

FarfieldPropagator::FarfieldPropagator() : CudaFunction("farfield_propagator")
{
}

void FarfieldPropagator::setParameters(size_t batch_size, size_t m, size_t n)
{
  batch_size_ = batch_size;
  m_ = m;
  n_ = n;
  sc_ =
      1.0f / std::sqrt(float(m * n));  // with cuFFT, we need to scale both ways
}

void FarfieldPropagator::setDeviceBuffers(complex<float> *d_datain,
                                          complex<float> *d_dataout,
                                          complex<float> *d_prefilter,
                                          complex<float> *d_postfilter)
{
  d_datain_ = d_datain;
  d_dataout_ = d_dataout;
  d_pre_ = d_prefilter;
  d_post_ = d_postfilter;
}

void FarfieldPropagator::allocate()
{
  {
    ScopedTimer t(this, "allocate");

    d_datain_.allocate(batch_size_ * m_ * n_);
    d_dataout_.allocate(batch_size_ * m_ * n_);
    d_pre_.allocate(m_ * n_);
    d_post_.allocate(m_ * n_);
  }

  {
    ScopedTimer t(this, "plan create");
    plan_ = FarfieldPropagator::planManager_.get_or_create_plan(
        batch_size_, m_, n_);
  }
}

void FarfieldPropagator::transfer_in(
    const complex<float> *data_to_be_transformed,
    const complex<float> *prefilter,
    const complex<float> *postfilter)
{
  ScopedTimer t(this, "transfer in");
  gpu_memcpy_h2d(
      d_datain_.get(), data_to_be_transformed, batch_size_ * m_ * n_);
  gpu_memcpy_h2d(d_pre_.get(), prefilter, m_ * n_);
  gpu_memcpy_h2d(d_post_.get(), postfilter, m_ * n_);
}

void FarfieldPropagator::transfer_out(complex<float> *out)
{
  ScopedTimer t(this, "transfer out");
  if (out)
  {
    gpu_memcpy_d2h(out, d_dataout_.get(), batch_size_ * m_ * n_);
  }
}

void FarfieldPropagator::run(bool doPreFilter,
                             bool doPostFilter,
                             bool isForward)
{
  ScopedTimer t(this, "run");

  size_t block = 256;
  size_t total = batch_size_ * m_ * n_;
  size_t blocks = (total + block - 1) / block;
  auto indata = d_datain_.get();
  if (doPreFilter)
  {
    applyPrefilter<<<blocks, block>>>(
        d_datain_.get(), d_dataout_.get(), d_pre_.get(), batch_size_, m_ * n_);
    checkLaunchErrors();
    indata = d_dataout_.get();
  }

  if (isForward)
  {
    checkCudaErrors(
        cufftExecC2C(plan_,
                     reinterpret_cast<cufftComplex *>(indata),
                     reinterpret_cast<cufftComplex *>(d_dataout_.get()),
                     CUFFT_FORWARD));
  }
  else
  {
    checkCudaErrors(
        cufftExecC2C(plan_,
                     reinterpret_cast<cufftComplex *>(indata),
                     reinterpret_cast<cufftComplex *>(d_dataout_.get()),
                     CUFFT_INVERSE));
  }

  if (doPostFilter)
  {
    applyPostfilter<<<blocks, block>>>(
        d_dataout_.get(), d_post_.get(), sc_, batch_size_, m_ * n_);
    checkLaunchErrors();
  }
  else
  {
    applyPostfilter<<<blocks, block>>>(
        d_dataout_.get(), nullptr, sc_, batch_size_, m_ * n_);
    checkLaunchErrors();
  }

  // sync device if timing is enabled
  timing_sync();
}

/************* Interface function ************/

extern "C" void farfield_propagator_c(const float *fdata_to_be_transformed,
                                      const float *fprefilter,
                                      const float *fpostfilter,
                                      float *fout,
                                      int b,
                                      int m,
                                      int n,
                                      int iisForward)
{
  auto data_to_be_transformed =
      reinterpret_cast<const complex<float> *>(fdata_to_be_transformed);
  // pre- and post-filter are 2D, applied in every batch item
  auto prefilter = reinterpret_cast<const complex<float> *>(fprefilter);
  auto postfilter = reinterpret_cast<const complex<float> *>(fpostfilter);
  auto out = reinterpret_cast<complex<float> *>(fout);
  auto isForward = iisForward != 0;

  auto prop = gpuManager.get_cuda_function<FarfieldPropagator>(
      "farfield_propagator", b, m, n);
  prop->allocate();
  prop->transfer_in(data_to_be_transformed, prefilter, postfilter);
  prop->run(prefilter != nullptr, postfilter != nullptr, isForward);
  prop->transfer_out(out);
}