#pragma once

#include "utils/CudaFunction.h"
#include "utils/Memory.h"
#include "utils/Complex.h"

/** Note: this function only does linear interpolation right so far. */
class InterpolatedShift : public CudaFunction {
public:
    InterpolatedShift();
    void setParameters(int items, int rows, int columns);
    void setDeviceBuffers(
        complex<float>* d_in,
        complex<float>* d_out
    );
    complex<float>* getOutput() const { return d_out_.get(); }
    void allocate();
    void transfer_in(
        const complex<float>* in
    );
    void run(float offsetRow, float offsetColumn, bool do_linear=false);
    void transfer_out(
        complex<float>* out
    );

private:
    DevicePtrWrapper<complex<float>> d_in_;
    DevicePtrWrapper<complex<float>> d_out_;
    int rows_ = 0, columns_ = 0, items_ = 0;
};
