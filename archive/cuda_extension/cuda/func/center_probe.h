#pragma once
#include "utils/Complex.h"
#include "utils/CudaFunction.h"
#include "utils/Memory.h"

#include "interpolated_shift.h"
#include "mass_center.h"

class CenterProbe : public CudaFunction
{
public:
  CenterProbe();
  void setParameters(int i, int m, int n);
  void setDeviceBuffers(complex<float>* d_probe, complex<float>* d_out);
  void allocate();
  void transfer_in(const complex<float>* probe);
  void run(float center_tolerance);
  void transfer_out(complex<float>* probe);

private:
  DevicePtrWrapper<complex<float>> d_probe_;
  DevicePtrWrapper<float> d_buffer_;
  DevicePtrWrapper<complex<float>> d_out_;
  int i_ = 0, m_ = 0, n_ = 0;
  InterpolatedShift* interp_shift_ = nullptr;
  MassCenter* mass_center_ = nullptr;
};
