#pragma once

#include "utils/Complex.h"
#include "utils/CudaFunction.h"
#include "utils/Memory.h"

class ScanAndMultiply : public CudaFunction
{
public:
  ScanAndMultiply(int batch_size,
                  int m,
                  int n,
                  int probe_i,
                  int probe_m,
                  int probe_n,
                  int obj_i,
                  int obj_m,
                  int obj_n);

  void setDeviceBuffers(complex<float> *d_probe,
                        complex<float> *d_obj,
                        int *d_addr_info,
                        complex<float> *d_out);
  void allocate();
  complex<float>* getOutput() const;
  void transfer_in(const complex<float> *probe,
                   const complex<float> *obj,
                   const int *addr_info);

  void transfer_out(complex<float> *out);

  void run();

private:
  DevicePtrWrapper<complex<float>> d_probe_;
  DevicePtrWrapper<complex<float>> d_obj_;
  DevicePtrWrapper<int> d_addr_info_;
  DevicePtrWrapper<complex<float>> d_out_;
  int batch_size_;
  int m_;
  int n_;
  int probe_i_, probe_m_, probe_n_, obj_i_, obj_m_, obj_n_;
};
