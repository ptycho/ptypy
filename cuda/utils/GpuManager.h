#pragma once

#include "utils/Memory.h"

#include <iostream>

/** Responsible for managing the GPUs globally.
 *
 * It creates the context and might eventually hold persistent 
 * CudaFunction objects, to keep memory allocated and re-usable.
 *
 * Don't create an instance directly - use the global object 
 * gpuManager instead.
 */
class GpuManager
{
public:
  GpuManager();
  int num_devices() const { return num_devices_; }
  ~GpuManager();

private:
  int num_devices_;
  DevicePtrWrapper<int> dummy_;
};

/** Global object to manage the GPUs */
extern GpuManager gpuManager;
