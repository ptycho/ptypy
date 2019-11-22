import numpy as np
from pycuda.compiler import SourceModule

from ..array_based import po_update_kernel as ab


class PoUpdateKernel(ab.PoUpdateKernel):

    def __init__(self, queue_thread=None):

        super(PoUpdateKernel, self).__init__(queue_thread)
        # and now initialise the cuda
        object_update_code = """
        #include <iostream>
        #include <utility>
        #include <thrust/complex.h>
        using thrust::complex;
        __device__ inline void atomicAdd(complex<float>* x, complex<float> y)
            {
              float* xf = reinterpret_cast<float*>(x);
              atomicAdd(xf, y.real());
              atomicAdd(xf + 1, y.imag());
            }
        
        extern "C"{
        __global__ void ob_update_cuda(
            const complex<float>* exit_wave,
            int A,
            int B,
            int C,
            const complex<float>* probe,
            int D,
            int E,
            int F,
            complex<float>* obj,
            int G,
            int H,
            int I,
            const int* addr,
            complex<float>* denominator
            )
            {
              int bid = blockIdx.x;
              int tx = threadIdx.x;
              int ty = threadIdx.y;
              int addr_stride = 15;
            
              const int* oa = addr + 3 + bid * addr_stride;
              const int* pa = addr + bid * addr_stride;
              const int* ea = addr + 6 + bid * addr_stride;
            
              probe += pa[0] * E * F + pa[1] * F + pa[2];
              obj += oa[0] * H * I + oa[1] * I + oa[2];
              denominator += oa[0] * H * I + oa[1] * I + oa[2];
            
              assert(oa[0] * H * I + oa[1] * I + oa[2] + (B - 1) * I + C - 1 < G * H * I);
            
              exit_wave += ea[0] * B * C;
            
              for (int b = tx; b < B; b += blockDim.x)
              {
                for (int c = ty; c < C; c += blockDim.y)
                {
                  atomicAdd(&obj[b * I + c], conj(probe[b * F + c]) * exit_wave[b * C + c] );
                  atomicAdd(&denominator[b * I + c], probe[b * F + c] * conj(probe[b * F + c]) );
                  }
               }
        }
        }
    
        """
        self.ob_update_cuda = SourceModule(object_update_code, include_dirs=[np.get_include()], no_extern_c=True).get_function("ob_update_cuda")

        probe_update_code = """
        #include <iostream>
        #include <utility>
        #include <thrust/complex.h>
        using thrust::complex;
        __device__ inline void atomicAdd(complex<float>* x, complex<float> y)
            {
              float* xf = reinterpret_cast<float*>(x);
              atomicAdd(xf, y.real());
              atomicAdd(xf + 1, y.imag());
            }
        
        extern "C"{
        __global__ void probe_update_cuda(
            const complex<float>* exit_wave,
            int A,
            int B,
            int C,
            complex<float>* probe,
            int D,
            int E,
            int F,
            const complex<float>* obj,
            int G,
            int H,
            int I,
            const int* addr,
            complex<float>* denominator
            )
            {
              int bid = blockIdx.x;
              int tx = threadIdx.x;
              int ty = threadIdx.y;
              int addr_stride = 15;
            
              const int* oa = addr + 3 + bid * addr_stride;
              const int* pa = addr + bid * addr_stride;
              const int* ea = addr + 6 + bid * addr_stride;
            
              probe += pa[0] * E * F + pa[1] * F + pa[2];
              obj += oa[0] * H * I + oa[1] * I + oa[2];
              denominator += pa[0] * E * F + pa[1] * F + pa[2];
            
              assert(oa[0] * H * I + oa[1] * I + oa[2] + (B - 1) * I + C - 1 < G * H * I);
            
              exit_wave += ea[0] * B * C;
            
              for (int b = tx; b < B; b += blockDim.x)
              {
                for (int c = ty; c < C; c += blockDim.y)
                {
                  atomicAdd(&probe[b * F + c], conj(obj[b * I + c]) * exit_wave[b * C + c] );
                  atomicAdd(&denominator[b * F + c], obj[b * I + c] * conj(obj[b * I + c]) );
                  }
               }
        }
        }

        """

        self.probe_update_cuda = SourceModule(probe_update_code, include_dirs=[np.get_include()],
                                       no_extern_c=True).get_function("probe_update_cuda")

    def ob_update(self, ob, obn, pr, ex, addr):
        self.ob_update_cuda(ex, self.num_pods, self.pr_shape[1], self.pr_shape[2],
                            pr, self.pr_shape[0], self.pr_shape[1], self.pr_shape[2],
                            ob, self.ob_shape[0], self.ob_shape[1], self.ob_shape[2],
                            addr,
                            obn,
                            block=(32, 32, 1), grid=(int(self.num_pods), 1, 1))

    def pr_update(self, pr, prn, ob, ex, addr):
        self.probe_update_cuda(ex, self.num_pods, self.pr_shape[1], self.pr_shape[2],
                               pr, self.pr_shape[0], self.pr_shape[1], self.pr_shape[2],
                               ob, self.ob_shape[0], self.ob_shape[1], self.ob_shape[2],
                               addr,
                               prn,
                               block=(32, 32, 1), grid=(int(self.num_pods), 1, 1))