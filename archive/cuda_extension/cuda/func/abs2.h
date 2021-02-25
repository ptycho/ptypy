#pragma once

#include "utils/CudaFunction.h"
#include "utils/Memory.h"

template <class Tin, class Tout>
class Abs2 : public CudaFunction
{
public:
  Abs2();
  void setParameters(size_t n);
  void setDeviceBuffers(Tin *d_datain, Tout *d_dataout);
  void allocate();
  Tout *getOutput() const;
  void transfer_in(const Tin *datain);
  void transfer_out(Tout *dataout);
  void run();

private:
  DevicePtrWrapper<Tin> d_datain_;
  DevicePtrWrapper<Tout> d_dataout_;
  size_t n_ = 0;
};