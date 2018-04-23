#include "GpuManager.h"
#include <cufft.h>
#include <iostream>

GpuManager::GpuManager()
{
  int numdev;
  cudaGetDeviceCount(&numdev);
  if (numdev == 0)
  {
    std::cout << "No GPUs found on this system\n";
    return;
  }

  properties_.resize(numdev);
  for (int i = 0; i < numdev; ++i)
  {
    cudaGetDeviceProperties(&properties_[i], i);
  }

  fftDummyHandles_.resize(numdev);
}

void GpuManager::selectDevice(int dev)
{
  if (dev >= int(properties_.size()))
  {
    throw GPUException("GPU " + std::to_string(dev) + " does not exist");
  }
  checkCudaErrors(cudaSetDevice(dev));
  if (selectedDevice_ == dev)
  {
    return;
  }
  selectedDevice_ = dev;

  // not initialised yet
  if (fftDummyHandles_[dev] == 0)
  {
    // init cufft and cuda context by creating a plan
    checkCudaErrors(cufftPlan1d(&fftDummyHandles_[dev], 32, CUFFT_C2C, 1));
  }
}

int GpuManager::getNumDevices() const { return int(properties_.size()); }

std::string GpuManager::getDeviceName(int id) const
{
  return properties_[id].name;
}

int GpuManager::getDeviceComputeCapability(int id) const
{
  return properties_[id].major * 10 + properties_[id].minor;
}

int GpuManager::getDeviceMemoryMB(int id) const
{
  return int(properties_[id].totalGlobalMem / 1024ul / 1024ul);
}

void GpuManager::resetFunctionCache() { functions_cache_.clear(); }

GpuManager::~GpuManager()
{
  for (int i = 0, ni = int(fftDummyHandles_.size()); i < ni; ++i)
  {
    if (fftDummyHandles_[i] != 0)
    {
      cudaSetDevice(i);
      cufftDestroy(fftDummyHandles_[i]);
      cudaDeviceReset();
    }
  }
}

// for memory tracking - see Memory.cpp
// needs to be here so that it gets destructed
// after the GpuManager object
std::map<void*, size_t> alloc_map_;
size_t alloc_total = 0;

// instatiate the object
GpuManager gpuManager;

/**** interface functions for python ******/

extern "C"
{
  int get_num_gpus_c() { return gpuManager.getNumDevices(); }

  int get_gpu_compute_capability_c(int dev)
  {
    return gpuManager.getDeviceComputeCapability(dev);
  }

  void select_gpu_device_c(int dev) { gpuManager.selectDevice(dev); }

  int get_gpu_memory_mb_c(int dev) { return gpuManager.getDeviceMemoryMB(dev); }

  std::string get_gpu_name_c(int dev) { return gpuManager.getDeviceName(dev); }

  void reset_function_cache_c() { gpuManager.resetFunctionCache(); }
}
