import numpy as np
from pycuda.compiler import SourceModule
from pycuda import gpuarray

from ..array_based import auxiliary_wave_kernel as ab


class AuxiliaryWaveKernel(ab.AuxiliaryWaveKernel):

    def __init__(self, queue_thread=None):
        super(AuxiliaryWaveKernel, self).__init__(queue_thread)
        # and now initialise the cuda
        build_aux_code = """
        #include <iostream>
        #include <utility>
        #include <thrust/complex.h>
        using thrust::complex;

        extern "C"{
        __global__ void build_aux_cuda(
            complex<float>* auxiliary_wave,
            const complex<float>* exit_wave,
            int A,
            int B,
            int C,
            const complex<float>* probe,
            int E,
            int F,
            const complex<float>* obj,
            int H,
            int I,
            const int* addr,
            float alpha
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
              exit_wave += ea[0] * B * C;
              auxiliary_wave += ea[0] * B * C;

              for (int b = tx; b < B; b += blockDim.x)
              {
                for (int c = ty; c < C; c += blockDim.y)
                {
                  auxiliary_wave[b * C + c] =  obj[b * I + c] * probe[b * F + c] * (1.0f + alpha) - exit_wave[b * C + c] * alpha;;
                  }
               }
        }
        }

        """
        self.build_aux_cuda = SourceModule(build_aux_code, include_dirs=[np.get_include()],
                                           no_extern_c=True).get_function("build_aux_cuda")

        build_exit_code = """
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
        __global__ void build_exit_cuda(
            complex<float>* auxiliary_wave,
            complex<float>* exit_wave,
            int A,
            int B,
            int C,
            const complex<float>* probe,
            int E,
            int F,
            const complex<float>* obj,
            int H,
            int I,
            const int* addr
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
              exit_wave += ea[0] * B * C;
              auxiliary_wave += ea[0] * B * C;

              for (int b = tx; b < B; b += blockDim.x)
              {
                for (int c = ty; c < C; c += blockDim.y)
                {
                  atomicAdd(&auxiliary_wave[b * C + c], probe[b * F + c] * obj[b * I + c] * -1.0f); // atomicSub is only for ints
                  atomicAdd(&exit_wave[b * C + c], auxiliary_wave[b * C + c] );
                  }
               }
        }
        }

        """

        self.build_exit_cuda = SourceModule(build_exit_code, include_dirs=[np.get_include()],
                                            no_extern_c=True).get_function("build_exit_cuda")

    def load(self, aux, ob, pr, ex, addr):
        super(AuxiliaryWaveKernel, self).load(aux, ob, pr, ex, addr)
        for key, array in self.npy.__dict__.items():
            self.ocl.__dict__[key] = gpuarray.to_gpu(array)

    def build_aux(self, auxiliary_wave, object_array, probe, exit_wave, addr):
        self.build_aux_cuda(auxiliary_wave,
                            exit_wave,
                            self.nmodes*self.nviews, np.int32(exit_wave.shape[1]), np.int32(exit_wave.shape[2]),
                            probe,
                            np.int32(exit_wave.shape[1]), np.int32(exit_wave.shape[2]),
                            object_array,
                            self.ob_shape[0], self.ob_shape[1],
                            addr,
                            self.alpha,
                            block=(32, 32, 1), grid=(int(self.nviews*self.nmodes), 1, 1), stream=self.queue)

    def build_exit(self, auxiliary_wave, object_array, probe, exit_wave, addr):
        self.build_exit_cuda(auxiliary_wave,
                             exit_wave,
                             self.nmodes*self.nviews, np.int32(exit_wave.shape[1]), np.int32(exit_wave.shape[2]),
                             probe,
                             np.int32(exit_wave.shape[1]), np.int32(exit_wave.shape[2]),
                             object_array,
                             self.ob_shape[0], self.ob_shape[1],
                             addr,
                             block=(32, 32, 1), grid=(int(self.nviews*self.nmodes), 1, 1), stream=self.queue)


