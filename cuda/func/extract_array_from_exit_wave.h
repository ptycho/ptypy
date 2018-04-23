#pragma once
#include "utils/Complex.h"
#include "utils/CudaFunction.h"
#include "utils/Memory.h"

class ExtractArrayFromExitWave : public CudaFunction
{
public:
  ExtractArrayFromExitWave();
  void setParameters(
      int A, int B, int C, int D, int E, int F, int G, int H, int I);
  void setDeviceBuffers(complex<float>* d_exit_wave,
                        int* d_exit_addr,
                        complex<float>* d_array_to_be_extracted,
                        int* d_extract_addr,
                        complex<float>* d_array_to_be_updated,
                        int* d_update_addr,
                        float* d_weights,
                        complex<float>* d_cfact,
                        complex<float>* d_denominator);
  void setAddrStride(int stride) { addr_stride_ = stride; }
  void allocate();
  void transfer_in(const complex<float>* exit_wave,
                   const int* exit_addr,
                   const complex<float>* array_to_be_extracted,
                   const int* extract_addr,
                   const complex<float>* array_to_be_updated,
                   const int* update_addr,
                   const float* weights,
                   const complex<float>* cfact);
  void run();
  void transfer_out(complex<float>* array_to_be_updated);

private:
  DevicePtrWrapper<complex<float>> d_exit_wave_;              // A x B x C
  DevicePtrWrapper<int> d_exit_addr_;                         // A x 3
  DevicePtrWrapper<complex<float>> d_array_to_be_extracted_;  // D x E x F
  DevicePtrWrapper<int> d_extract_addr_;                      // A x 3
  DevicePtrWrapper<complex<float>> d_array_to_be_updated_;    // G x H x I
  DevicePtrWrapper<int> d_update_addr_;                       // A x 3
  DevicePtrWrapper<float> d_weights_;                         // G
  DevicePtrWrapper<complex<float>> d_cfact_;                  // G x H x I
  DevicePtrWrapper<complex<float>> d_denominator_;            // G x H x I
  int A_ = 0, B_ = 0, C_ = 0, D_ = 0, E_ = 0, F_ = 0, G_ = 0, H_ = 0, I_ = 0;
  int addr_stride_ = 3;  // default is 3, but if full addr_info is used, it's 15
};