#include <iostream>
#include <utility>
#include <thrust/complex.h>
#include <stdio.h>
using thrust::complex;
using std::sqrt;
using thrust::abs;

extern "C"{
__global__ void 
__launch_bounds__(1024, 2)
fourier_error(int nmodes,
                           complex<float> *f,
                           const float *fmask,
                           const float *fmag,
                           float *fdev,
                           float *ferr,
                           const float * mask_sum,
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

      for (int a = ty; a < A; a += blockDim.y)
      {
        for (int b = tx; b < B; b += blockDim.x)
        {
          float acc = 0.0;
          for (int idx = 0; idx < nmodes; ++idx )
          {
           float abs_exit_wave = abs(f[a * B + b + idx*A*B]);
           acc += abs_exit_wave * abs_exit_wave; // if we do this manually (real*real +imag*imag) we get bad rounding errors
          }
          auto fdevv = sqrt(acc) - fmag[a * B + b];
          ferr[a * B + b] = (fmask[a * B + b] * fdevv * fdevv) / mask_sum[ma[0]];
          fdev[a * B + b] = fdevv;
        }
      }

}
}