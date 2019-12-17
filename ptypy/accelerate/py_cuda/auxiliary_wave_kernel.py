import numpy as np

from pycuda import gpuarray
from . import load_kernel

from ..array_based import auxiliary_wave_kernel as ab



class AuxiliaryWaveKernel(ab.AuxiliaryWaveKernel):

    def __init__(self, queue_thread=None):
        super(AuxiliaryWaveKernel, self).__init__()
        # and now initialise the cuda
        self._ob_shape = None
        self._ob_id = None
        self.build_aux_cuda = load_kernel("build_aux")
        self.build_exit_cuda = load_kernel("build_exit")

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
