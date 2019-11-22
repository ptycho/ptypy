import numpy as np
from pycuda.compiler import SourceModule

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
            int D,
            int E,
            int F,
            const complex<float>* obj,
            int G,
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

    def build_aux(self, auxiliary_wave, object_array, probe, exit_wave, addr):
        self.build_aux_cuda(auxiliary_wave,
                            exit_wave,
                            self.num_pods, self.pr_shape[1], self.pr_shape[2],
                            probe,
                            self.pr_shape[0], self.pr_shape[1], self.pr_shape[2],
                            object_array,
                            self.ob_shape[0], self.ob_shape[1], self.ob_shape[2],
                            addr,
                            self.alpha,
                            block=(32, 32, 1), grid=(int(self.num_pods), 1, 1))

    def build_exit(self, aux, ob, pr, ex, addr):
        raise NotImplementedError('build_exit is not yet implemented')



