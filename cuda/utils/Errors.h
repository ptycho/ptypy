#pragma once

#include "utils/Patches.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <string>

/** GPU exceptions - typically thrown from CUDA API failures.
 *
 * Using the checkCudaErrors() macro around each call makes sure
 * that this exception is thrown on error.
 */
class GPUException : public std::runtime_error
{
public:
  GPUException(int code,
               const std::string &what,
               const std::string &instruction = "<unknown>",
               const std::string &file = "<unknown>",
               int line = -1)
      : std::runtime_error("GPU Exception at " + file + ":" +
                           std::to_string(line) + ": " + what +
                           " @ instruction '" + instruction + "'")
  {
  }
  GPUException(const std::string &what) : std::runtime_error(what) {}
};

/// Wrap this around cuda or cufft API calls
#define checkCudaErrors(val)                                                   \
  detail::handleCudaErrors((val), #val, __FILE__, __LINE__)

/// Use this macro to check if a kernel launch failed.
/// That is, call this straight after a manual kernel launch, without sync.
#define checkLaunchErrors() checkCudaErrors(cudaPeekAtLastError())

/// Implementation details, translating macro to exception
namespace detail
{
inline void handleCudaErrors(cudaError_t e,
                             const char *func,
                             const char *file,
                             int line)
{
  if (e)
  {
    cudaDeviceReset();
    throw GPUException(int(e), cudaGetErrorString(e), func, file, line);
  }
}

inline void handleCudaErrors(cufftResult e,
                             const char *func,
                             const char *file,
                             int line)
{
  if (e)
  {
    const char *errstr = nullptr;
    switch (e)
    {
      case CUFFT_INVALID_PLAN:
        errstr = "cuFFT was passed an invalid plan handle";
        break;
      case CUFFT_ALLOC_FAILED:
        errstr = "cuFFT failed to allocate GPU or CPU memory";
        break;
      case CUFFT_INVALID_TYPE:
        errstr = "No longer used";
        break;
      case CUFFT_INVALID_VALUE:
        errstr = "User specified an invalid pointer or parameter";
        break;
      case CUFFT_INTERNAL_ERROR:
        errstr = "Driver or internal cuFFT library error";
        break;
      case CUFFT_EXEC_FAILED:
        errstr = "Failed to execute an FFT on the GPU";
        break;
      case CUFFT_SETUP_FAILED:
        errstr = "The cuFFT library failed to initialize";
        break;
      case CUFFT_INVALID_SIZE:
        errstr = "User specified an invalid transform size";
        break;
      case CUFFT_UNALIGNED_DATA:
        errstr = " No longer used";
        break;
      case CUFFT_INCOMPLETE_PARAMETER_LIST:
        errstr = " Missing parameters in call ";
        break;
      case CUFFT_INVALID_DEVICE:
        errstr = "Execution of a plan was on different GPU than plan creation";
        break;
      case CUFFT_PARSE_ERROR:
        errstr = "Internal plan database error";
        break;
      case CUFFT_NO_WORKSPACE:
        errstr = "No workspace has been provided prior to plan execution";
        break;
      case CUFFT_NOT_IMPLEMENTED:
        errstr =
            "Function does not implement functionality for parameters given.";
        break;
      case CUFFT_LICENSE_ERROR:
        errstr = "Used in previous versions.";
        break;
      case CUFFT_NOT_SUPPORTED:
        errstr = "Operation is not supported for parameters given.";
        break;
      default:
        errstr = "Unknown";
    }

    cudaDeviceReset();
    throw GPUException(int(e), errstr, func, file, line);
  }
}

}  // namespace detail
