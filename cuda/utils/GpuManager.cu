#include "GpuManager.h"
#include <iostream>
#include <cufft.h>


static cufftHandle fftplan;

GpuManager::GpuManager()
{
  // initialize Context
  cudaGetDeviceCount(&num_devices_);
  std::cout << "Found " << num_devices_ << " GPUs" << std::endl;

  checkCudaErrors(cudaDeviceSynchronize());
  
  // init cufft and cuda context by creating a plan, this is completely unused, but just forces handle creation here.
  int dims[] = {int(256), int(256)};
  size_t workSize;
  checkCudaErrors(cufftCreate(&fftplan));
  checkCudaErrors(cufftMakePlanMany(
        fftplan, 2, dims, 0, 0, 0, 0, 0, 0, CUFFT_C2C, 10, &workSize));
}

GpuManager::~GpuManager() {
    cufftDestroy(fftplan);
    cudaDeviceReset();
}


// instatiate the object
GpuManager gpuManager;
