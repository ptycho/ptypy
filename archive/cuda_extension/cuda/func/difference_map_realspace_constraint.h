#pragma once

#include "utils/Complex.h"
#include "utils/CudaFunction.h"
#include "utils/Memory.h"

class DifferenceMapRealspaceConstraint : public CudaFunction
{
public:
  DifferenceMapRealspaceConstraint();
  void setParameters(int i, int m, int n);
  void setDeviceBuffers(complex<float> *d_obj_and_probe,
                        complex<float> *d_exit_wave,
                        complex<float> *d_out);
  void allocate();
  complex<float> *getOutput() const;
  void transfer_in(const complex<float> *obj_and_probe,
                   const complex<float> *exit_wave);
  void transfer_out(complex<float> *out);
  void run(float alpha);

private:
  DevicePtrWrapper<complex<float>> d_obj_and_probe_;
  DevicePtrWrapper<complex<float>> d_exit_wave_;
  DevicePtrWrapper<complex<float>> d_out_;
  int i_ = 0, m_ = 0, n_ = 0;
};
