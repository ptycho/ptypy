#pragma once
#include "utils/Complex.h"
#include "utils/CudaFunction.h"
#include "utils/Memory.h"

class GetDifference : public CudaFunction
{
public:
  GetDifference();
  void setParameters(int i, int m, int n);
  void setDeviceBuffers(int *d_addr_info,
                        complex<float> *d_backpropagated_solution,
                        float *d_err_fmag,
                        complex<float> *d_exit_wave,
                        complex<float> *d_probe_obj,
                        complex<float> *d_out);
  void allocate();
  void updateErrorInput(float *d_err_fmag);
  complex<float> *getOutput() const;
  void transfer_in(const int *addr_info,
                   const complex<float> *backpropagated_solution,
                   const float *err_fmag,
                   const complex<float> *exit_wave,
                   const complex<float> *probe_obj);
  void run(float alpha, float pbound = 0.0f, bool usePbound = false);
  void transfer_out(complex<float> *out);

private:
  DevicePtrWrapper<int> d_addr_info_;
  DevicePtrWrapper<complex<float>> d_backpropagated_solution_;
  DevicePtrWrapper<float> d_err_fmag_;
  DevicePtrWrapper<complex<float>> d_exit_wave_;
  DevicePtrWrapper<complex<float>> d_probe_obj_;
  DevicePtrWrapper<complex<float>> d_out_;
  int i_ = 0, m_ = 0, n_ = 0;
};