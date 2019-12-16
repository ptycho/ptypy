import numpy as np
from pycuda.compiler import SourceModule
from inspect import getfullargspec

from ..array_based import fourier_update_kernel as ab
from pycuda import gpuarray


class FourierUpdateKernel(ab.FourierUpdateKernel):

    def __init__(self, queue_thread=None, nmodes=1, pbound=0.0):
        super(FourierUpdateKernel, self).__init__(queue_thread=queue_thread, nmodes=nmodes, pbound=pbound)
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

        fourier_error_code = """
        #include <iostream>
        #include <utility>
        #include <thrust/complex.h>
        #include <stdio.h>
        using thrust::complex;
        using std::sqrt;
        using thrust::abs;

        extern "C"{
        __global__ void fourier_error_cuda(int nmodes,
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
        """
        self.fourier_error_cuda = SourceModule(fourier_error_code, include_dirs=[np.get_include()],
                                               no_extern_c=True).get_function("fourier_error_cuda")

        err_reduce_code = """
        #include <iostream>
        #include <utility>
        #include <thrust/complex.h>
        #include <stdio.h>


        extern "C"{
        __global__ void error_reduce_cuda(float *ferr,
                                          float *err_fmag,
                                          int M,
                                          int N)
        {
          int tx = threadIdx.x;
          int ty = threadIdx.y;
          int batch = blockIdx.x;
          extern __shared__ float sum_v[];

          int shidx = tx * blockDim.y + ty; // shidx is the index in shared memory for this single block
          sum_v[shidx] = 0.0;

          for (int m = tx; m < M; m += blockDim.x)
          {
            for (int n = ty; n < N; n += blockDim.y)
            {
              int idx = batch * M * N + m * N + n; // idx is index qwith respect to the full stack
              sum_v[shidx] += ferr[idx];
            }
          }


          __syncthreads();
          int nt = blockDim.x * blockDim.y;
          int c = nt;

          while (c > 1)
          {
                int half = c / 2;
                if (shidx < half)
                {
                  sum_v[shidx] += sum_v[c - shidx - 1];
                }
                __syncthreads();
                c = c - half;
          }

          if (shidx == 0)
          {
            err_fmag[batch] = float(sum_v[0]);
          }
          __syncthreads();
        }
        }
        """
        self.error_reduce_cuda = SourceModule(err_reduce_code, include_dirs=[np.get_include()],
                                              no_extern_c=True).get_function("error_reduce_cuda")

    def configure(self, I, mask, f, addr):
        super(FourierUpdateKernel, self).configure(I, mask, f , addr)
        for key, array in self.npy.__dict__.items():
            self.ocl.__dict__[key] = gpuarray.to_gpu(array)

    def fourier_error(self, f, fmag, fdev, ferr, fmask, mask_sum, addr):
        #print(self.fshape)
        self.fourier_error_cuda(np.int32(self.nmodes),
                                f,
                                fmask,
                                fmag,
                                fdev,
                                ferr,
                                mask_sum,
                                addr,
                                np.int32(self.fshape[1]),
                                np.int32(self.fshape[2]),
                                block=(32, 32, 1),
                                grid=(int(self.fshape[0]), 1, 1),
                                stream=self.queue)

    def error_reduce(self, ferr, err_fmag, addr):
        import sys
        float_size = sys.getsizeof(np.float32(4))
        # shared_memory_size =int(2 * 32 * 32 *float_size) # this doesn't work even though its the same...
        shared_memory_size = int(49152)

        self.error_reduce_cuda(ferr,
                               err_fmag,
                               np.int32(self.fshape[1]),
                               np.int32(self.fshape[2]),
                               block=(32, 32, 1),
                               grid=(int(self.fshape[0]), 1, 1),
                               shared=shared_memory_size,
                               stream=self.queue)

    def calc_fm(self, fm, fmask, fmag, fdev, err_fmag, addr):
        raise NotImplementedError('The calc_fm kernel is not implemented yet')

    def fmag_update(self, f, fm, addr):
        raise NotImplementedError('The fmag_update kernel is not implemented yet')

    def fmag_all_update(self, f, fmask, fmag, fdev, err_fmag, addr):
        self.fmag_all_update_cuda(f,
                                  fmask,
                                  fmag,
                                  fdev,
                                  err_fmag,
                                  addr,
                                  np.float32(self.pbound),
                                  np.int32(self.fshape[1]),
                                  np.int32(self.fshape[2]),
                                  block=(32, 32, 1),
                                  grid=(int(self.fshape[0]*self.nmodes), 1, 1),
                                  stream=self.queue)

    def execute(self, kernel_name=None, compare=False, sync=False):

        if kernel_name is None:
            for kernel in self.kernels:
                self.execute(kernel, compare, sync)
        else:
            self.log("KERNEL " + kernel_name)
            meth = getattr(self, kernel_name)
            kernel_args = getfullargspec(meth).args[1:]
            args = [getattr(self.ocl, a) for a in kernel_args]
            meth(*args)

        return self.ocl.err_fmag.get()