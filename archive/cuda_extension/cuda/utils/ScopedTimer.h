#pragma once

#include "utils/CudaFunction.h"
#include "utils/Errors.h"
#include "utils/Timer.h"
#include <string>

/** If we do timing, this function syncs all device execution code first.
 *
 * Otherwise it doesn't do anything
 */
inline void timing_sync()
{
#if DO_GPU_TIMING
  checkCudaErrors(cudaDeviceSynchronize());
#endif
}

/** Scoped RAII-style timer class, that records times from construction
 *  to destruction and logs it in a CudaFunction object.
 */
class ScopedTimer
{
public:
  /** Constructor. Starts recording time immediately.
   *
   * @param func CudaFunction object to store the resulting time in.
   * @param name Name of the operation that is timed.
   */
  ScopedTimer(CudaFunction *func, const std::string &name)
      : func_(func), name_(name)
  {
  }

  /** Stops timing and records the time in the CudaFunction given
   *  to the constructor (only if DO_GPU_TIMING is defined).
   */
  ~ScopedTimer()
  {
#if DO_GPU_TIMING
    func_->times_.emplace_back(name_, t_.get_time());
#endif
  }

private:
#if DO_GPU_TIMING
  Timer t_;
#endif
  CudaFunction *func_;
  std::string name_;
};