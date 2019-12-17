import numpy as np
from . import load_kernel
from inspect import getfullargspec
from ..array_based import fourier_update_kernel as ab
from pycuda import gpuarray


class FourierUpdateKernel(ab.FourierUpdateKernel):

    def __init__(self, aux, nmodes=1, queue_thread=None):
        super(FourierUpdateKernel, self).__init__(aux,  nmodes=nmodes)

        self.fmag_all_update_cuda = load_kernel("fmag_all_update")
        self.fourier_error_cuda = load_kernel("fourier_error")
        self.error_reduce_cuda = load_kernel("error_reduce")

    def allocate(self):
        self.npy.fdev = gpuarray.zeros(self.fshape, dtype=np.float32)
        self.npy.ferr = gpuarray.zeros(self.fshape, dtype=np.float32)

    def fourier_error(self, f, addr, fmag, fmask, mask_sum):
        fdev = self.npy.fdev
        ferr = self.npy.ferr
        # print(self.fshape)
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
                                grid=(int(fmag.shape[0]), 1, 1),
                                    stream=self.queue)

    def error_reduce(self, addr, err_fmag):
        import sys
        float_size = sys.getsizeof(np.float32(4))
        # shared_memory_size =int(2 * 32 * 32 *float_size) # this doesn't work even though its the same...
        shared_memory_size = int(49152)

        self.error_reduce_cuda(self.npy.ferr,
                               err_fmag,
                               np.int32(self.fshape[1]),
                               np.int32(self.fshape[2]),
                               block=(32, 32, 1),
                               grid=(int(err_fmag.shape[0]), 1, 1),
                               shared=shared_memory_size,
                               stream=self.queue)

    def calc_fm(self, fm, fmask, fmag, fdev, err_fmag, addr):
        raise NotImplementedError('The calc_fm kernel is not implemented yet')

    def fmag_update(self, f, fm, addr):
        raise NotImplementedError('The fmag_update kernel is not implemented yet')

    def fmag_all_update(self, f, addr, fmag, fmask, err_fmag, pbound=0.0):
        sh = fmag.shape
        fdev = self.npy.fdev
        sh = fmag.shape
        self.fmag_all_update_cuda(f,
                                  fmask,
                                  fmag,
                                  fdev,
                                  err_fmag,
                                  addr,
                                  np.float32(pbound),
                                  np.int32(self.fshape[1]),
                                  np.int32(self.fshape[2]),
                                  block=(32, 32, 1),
                                  grid=(int(fmag.shape[0]*self.nmodes), 1, 1),
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