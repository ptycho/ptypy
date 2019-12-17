#include <iostream>
#include <utility>
#include <thrust/complex.h>
#include <stdio.h>
using thrust::complex;
using std::sqrt;

extern "C"{
__global__ void fmag_all_update(complex<float> *f,
                                     const float *fmask,
                                     const float *fmag,
                                     const float *fdev,
                                     const float *err_fmag,
                                     const int *addr_info,
                                     float pbound,
                                     int A,
                                     int B)
    {
      int batch = blockIdx.x;
      int tx = threadIdx.x;
      int ty = threadIdx.y;
      int addr_stride = 15;

      const int* ea = addr_info + batch * addr_stride + 6;
      const int* da = addr_info + batch * addr_stride + 9;
      const int* ma = addr_info + batch * addr_stride + 12;

      fmask += ma[0] * A * B ;
      float err = err_fmag[da[0]];
      fdev += da[0] * A * B ;
      fmag += da[0] * A * B ;
      f += ea[0] * A * B ;
      float renorm = sqrt(pbound / err);

      for (int a = tx; a < A; a += blockDim.x)
      {
        for (int b = ty; b < B; b += blockDim.y)
        {
          float m = fmask[a * A + b];
          if (renorm < 1.0f)
          {

            float fm = (1.0f - m) + m * ((fmag[a * A + b] + fdev[a * A + b] * renorm) / (fdev[a * A + b] + fmag[a * A + b]  + 1e-10f)) ;
            f[a * A + b] = fm * f[a * A + b];
          }

      }
    }
}
}