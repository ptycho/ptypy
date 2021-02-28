#include <cmath>
#include <thrust/complex.h>
using std::sqrt;
using thrust::complex;

extern "C" __global__ void fmag_all_update(complex<float>* f,
                                           const float* fmask,
                                           const float* fmag,
                                           const float* fdev,
                                           const float* err_fmag,
                                           const int* addr_info,
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

  fmask += ma[0] * A * B;
  float err = err_fmag[da[0]];
  fdev += da[0] * A * B;
  fmag += da[0] * A * B;
  f += ea[0] * A * B;
  float renorm = sqrt(pbound / err);

  for (int a = ty; a < A; a += blockDim.y)
  {
    for (int b = tx; b < B; b += blockDim.x)
    {
      float m = fmask[a * A + b];
      if (renorm < 1.0f)
      {
        /*
        // assuming this is actually a mask, i.e. 0 or 1 --> this is slower
        float fm = m < 0.5f ? 1.0f :
          ((fmag[a * A + b] + fdev[a * A + b] * renorm) / (fdev[a * A + b] +
        fmag[a * A + b]  + 1e-7f)) ;
        */
        auto fmagv = fmag[a * A + b];
        auto fdevv = fdev[a * A + b];
        float fm = (1.0f - m) +
                   m * ((fmagv + fdevv * renorm) / (fmagv + fdevv + 1e-7f));
        f[a * A + b] *= fm;
      }
    }
  }
}
