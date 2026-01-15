#pragma once
#include "utils/Complex.h"
#include "utils/CudaFunction.h"
#include "utils/Memory.h"

class ComplexGaussianFilter : public CudaFunction
{
public:
  // number of standard deviations to use for the filter
  static const int NUM_STDDEVS = 4;

  static const int ROWS_BLOCKDIM_X = 16;  // width of tile - match kernel size?
  static const int ROWS_BLOCKDIM_Y = 4;   // height of tile
  static const int MAX_SHARED_PER_BLOCK =
      48 * 1024 / 2;  // at least 2 blocks per SM
  static const int MAX_SHARED_PER_BLOCK_COMPLEX =
      MAX_SHARED_PER_BLOCK / 2 * sizeof(float);
  static const int MAX_KERNEL_RADIUS =
      MAX_SHARED_PER_BLOCK_COMPLEX / ROWS_BLOCKDIM_X;

  ComplexGaussianFilter();
  void setParameters(int ndims, const int* shape, const float* mfs);
  void setDeviceBuffers(complex<float>* d_input, complex<float>* d_output);
  void allocate();
  void transfer_in(const complex<float>* input);
  void run();
  void transfer_out(complex<float>* output);

private:
  int totalSize() const;
  std::vector<float> calcConvolutionKernel(float stddev, int ndevs);

  DevicePtrWrapper<complex<float>> d_input_;
  DevicePtrWrapper<complex<float>> d_output_;
  int ndims_ = 2;
  int shape_[3];
  float mfs_[3];
};
