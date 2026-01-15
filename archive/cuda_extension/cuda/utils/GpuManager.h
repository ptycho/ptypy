#pragma once

#include "utils/Complex.h"
#include "utils/Memory.h"

#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "utils/CudaFunction.h"
#include <cuda_runtime.h>
#include <cufft.h>

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
  /** Queries available devices and their properties, but doesn't initialise the
   * context
   */
  GpuManager();

  /** Get number of available GPUs.
   *
   * Note the Cuda GPUs are numbered with integer indices, always continuous.
   */
  int getNumDevices() const;

  /** Get the name of a specific GPU */
  std::string getDeviceName(int id) const;

  /** Get the compute capability of a specific GPU, in format 35 for 3.5. */
  int getDeviceComputeCapability(int id) const;

  /** Get global memory available in MB */
  int getDeviceMemoryMB(int id) const;

  /** Select a specific device to use for all calculations that follow.
   *
   * Note that this function creates the context on the given device,
   * to avoid setup delays. Also, this device should not be changed once
   * functions have been executed already. A resetFunctionCache call is needed
   * before in this case.
   */
  void selectDevice(int dev = 0);

  /** Resets (clears) the internal cache for CudaFunction objects
   *
   * Subsequent calls to CudaFunctions will re-create them, allocating new
   * memory, etc.
   */
  void resetFunctionCache();

  /** Function to obtain an instance of a specific CudaFunction, given its name
   * key and parameters.
   *
   * This functions looks up in the cache if an object with the given name
   * already exists. The name is used as a key in the cache (so it should be
   * unique for each distinct CudaFunction type).
   *
   * If the function isn't in the cache, it is created.
   *
   * After that, it calls setParameters on the CudaFunction, passing the
   * variable parameters given to this function after the name key.
   *
   * Therefore all CudaFunction objects created with this functions must:
   * - derive from the CudaFunction base class
   * - have a default constructor
   * - have a setParameters() method which accepts the arguments passed this
   *   this method.
   * - should be written so that it respects updated setParameters in subsquent
   *   calls to allocate, transfer_in, setDeviceBuffers, etc... 
   */
  template <typename T, typename... Args>
  T* get_cuda_function(const std::string& key, Args&&... args)
  {
    // setup device if not already done
    if (selectedDevice_ == -1)
    {
      selectDevice(0);
    }

    // create if not already there (with default constructor)
    if (functions_cache_.find(key) == functions_cache_.end())
    {
      functions_cache_[key] = std::unique_ptr<T>(new T());
    }

    // retrieve and set the parameters (they might have changed
    // since we cached them)
    auto ret = dynamic_cast<T*>(functions_cache_[key].get());
    if (ret == nullptr)
    {
      throw GPUException("cached CudaFunction with key " + key +
                         " doesn't match the requested type");
    }
    ret->setParameters(std::forward<Args>(args)...);

    return ret;
  }

  ~GpuManager();

private:
  std::vector<cudaDeviceProp> properties_;
  std::vector<cufftHandle> fftDummyHandles_;
  int selectedDevice_ = -1;
  std::map<std::string, std::unique_ptr<CudaFunction>> functions_cache_;
};

/** Global object to manage the GPUs */
extern GpuManager gpuManager;

/** Type helper - converting type signature into a string.
 *
 * This is used for generating a signature key to use in the function_cache
 * above - to avoid for example Abs2<float> and Abs2<double> to have the same
 * key in the cache.
 **/
template <class T>
inline std::string getTypeName()
{
  return "unknown";
}
template <>
inline std::string getTypeName<float>()
{
  return "float";
}
template <>
inline std::string getTypeName<double>()
{
  return "double";
}
template <>
inline std::string getTypeName<complex<float>>()
{
  return "c_float";
}
template <>
inline std::string getTypeName<complex<double>>()
{
  return "c_double";
}

/**** interface functions for python ******/
extern "C"
{
  int get_num_gpus_c();
  int get_gpu_compute_capability_c(int dev);
  void select_gpu_device_c(int dev);
  int get_gpu_memory_mb_c(int dev);
  std::string get_gpu_name_c(int dev);
  void reset_function_cache_c();
}