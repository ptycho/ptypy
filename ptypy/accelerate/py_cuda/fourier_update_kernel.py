import numpy as np
from pycuda.compiler import SourceModule

from ..array_based import fourier_update_kernel as ab


class FourierUpdateKernel(ab.FourierUpdateKernel):

    def __init__(self, queue_thread=None):

        super(FourierUpdateKernel, self).__init__(queue_thread)
        fmag_all_update_cuda_code = """
        #include <iostream>
        #include <utility>
        #include <thrust/complex.h>
        #include <stdio.h>
        using thrust::complex;
        using std::sqrt;
        
        extern "C"{
        __global__ void fmag_all_update_cuda(complex<float> *f,
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
        """
        self.fmag_all_update_cuda = SourceModule(fmag_all_update_cuda_code, include_dirs=[np.get_include()],
                                           no_extern_c=True).get_function("fmag_all_update_cuda")

    def fourier_error(self, f, fmag, fdev, ferr, fmask):
        raise NotImplementedError('The fourier error kernel is not implemented yet')

    def error_reduce(self, ferr, err_fmag):
        raise NotImplementedError('The error_reduce kernel is not implemented yet')

    def calc_fm(self, fm, fmask, fmag, fdev, err_fmag):
        raise NotImplementedError('The calc_fm kernel is not implemented yet')

    def fmag_update(self, f, fm):
        raise NotImplementedError('The fmag_update kernel is not implemented yet')

    def fmag_all_update(self, f, fmask, fmag, fdev, err_fmag, addr_info):
        self.fmag_all_update_cuda(f,
                                  fmask,
                                  fmag,
                                  fdev,
                                  err_fmag,
                                  addr_info,
                                  self.pbound,
                                  self.f_shape[1],
                                  self.f_shape[2],
                                  block=(32, 32, 1),
                                  grid=(int(self.num_pods), 1, 1))
