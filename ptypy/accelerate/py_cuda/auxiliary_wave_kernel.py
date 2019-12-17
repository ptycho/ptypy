import numpy as np
from pycuda.compiler import SourceModule
from pycuda import gpuarray

from ..array_based import auxiliary_wave_kernel as ab


class AuxiliaryWaveKernel(ab.AuxiliaryWaveKernel):

    def __init__(self, queue_thread=None):
        super(AuxiliaryWaveKernel, self).__init__()
        # and now initialise the cuda
        self._ob_shape = None
        self._ob_id = None

        build_aux_code = """
        #include <iostream>
        #include <utility>
        #include <thrust/complex.h>
        using thrust::complex;

        extern "C"{
        __global__ void build_aux_cuda(
            complex<float>* auxiliary_wave,
            const complex<float>* exit_wave,
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

    def build_aux(self, b_aux, addr, ob, pr, ex, alpha):
        obr, obc = self._cache_object_shape(ob)
        self.build_aux_cuda(b_aux,
                            ex,
                            np.int32(ex.shape[1]), np.int32(ex.shape[2]),
                            pr,
                            np.int32(ex.shape[1]), np.int32(ex.shape[2]),
                            ob,
                            obr, obc,
                            addr,
                            alpha,
                            block=(32, 32, 1), grid=(int(ex.shape[0]), 1, 1), stream=self.queue)

    def build_exit(self, b_aux, addr, ob, pr, ex):
        obr, obc = self._cache_object_shape(ob)
        self.build_exit_cuda(b_aux,
                             ex,
                             np.int32(ex.shape[1]), np.int32(ex.shape[2]),
                             pr,
                             np.int32(ex.shape[1]), np.int32(ex.shape[2]),
                             ob,
                             obr, obc,
                             addr,
                             block=(32, 32, 1), grid=(int(ex.shape[0]), 1, 1), stream=self.queue)

    def _cache_object_shape(self, ob):
        oid = id(ob)

        if not oid == self._ob_id:
            self._ob_id = oid
            self._ob_shape = (np.int32(ob.shape[-2]), np.int32(ob.shape[-1]))

        return self._ob_shape
