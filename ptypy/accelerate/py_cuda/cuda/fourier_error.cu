#include <iostream>
#include <utility>
#include <thrust/complex.h>
#include <stdio.h>
using thrust::complex;
using std::sqrt;
using thrust::abs;

extern "C"{
__global__ void fourier_error(int nmodes,
                           complex<float> *f,
                           const float *fmask,
                           const float *fmag,
                           float *fdev,
                           float *ferr,
                           const float *mask_sum,
                           const int *addr,
                           int A,
                           int B
                           )
{
      int tx = threadIdx.x;
      int ty = threadIdx.y;
      int addr_stride = 15;

      const int* ea = addr + 6 + (blockIdx.x * nmodes) * addr_stride;
      const int* da = addr + 9 + (blockIdx.x * nmodes) * addr_stride;
      const int* ma = addr + 12 + (blockIdx.x * nmodes) * addr_stride;

      f += ea[0] * A * B;
      fdev += da[0] * A * B;
      fmag += da[0] * A * B;
      fmask += ma[0] * A * B;
      ferr += da[0] * A * B;

      for (int a = tx; a < A; a += blockDim.x)
      {
        for (int b = ty; b < B; b += blockDim.y)
        {
          float acc = 0.0;
          for (int idx = 0; idx < nmodes; idx+=1 )
          {
           float abs_exit_wave = abs(f[a * B + b + idx*A*B]);
           acc += abs_exit_wave * abs_exit_wave; // if we do this manually (real*real +imag*imag) we get bad rounding errors
          }
          fdev[a * B + b] = sqrt(acc) - fmag[a * B + b];
          float abs_fdev = abs(fdev[a * B + b]);
          ferr[a * B + b] = (fmask[a * B + b] * abs_fdev * abs_fdev) / mask_sum[ma[0]];
        }
      }

}
}