#pragma once

#include "utils/Complex.h"
#include "utils/CudaFunction.h"
#include "utils/Memory.h"

/** Clips the complex magnitudes to given range
 *
 * BEWARE - this function works on the input in-place
 */
class ClipComplexMagnitudesToRange : public CudaFunction
{
public:
  ClipComplexMagnitudesToRange();
  void setParameters(int n);
  void setDeviceBuffers(complex<float>* d_data);
  void allocate();
  void transfer_in(const complex<float>* data);
  void run(float clip_min, float clip_max);
  void transfer_out(complex<float>* data);

private:
  DevicePtrWrapper<complex<float>> d_data_;
  int n_ = 0;
};