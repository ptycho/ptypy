import numpy as np
from inspect import getfullargspec
from pycuda import gpuarray
from ptypy.utils.verbose import log
from . import load_kernel
from ..array_based import kernels as ab


class FourierUpdateKernel(ab.FourierUpdateKernel):

    def __init__(self, aux, nmodes=1, queue_thread=None):
        super(FourierUpdateKernel, self).__init__(aux,  nmodes=nmodes)
        self.queue = queue_thread
        self.fmag_all_update_cuda = load_kernel("fmag_all_update")
        self.fourier_error_cuda = load_kernel("fourier_error")
        self.error_reduce_cuda = load_kernel("error_reduce")

    def allocate(self):
        self.npy.fdev = gpuarray.zeros(self.fshape, dtype=np.float32)
        self.npy.ferr = gpuarray.zeros(self.fshape, dtype=np.float32)

    def fourier_error(self, f, addr, fmag, fmask, mask_sum):
        fdev = self.npy.fdev
        ferr = self.npy.ferr
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
        # float_size = sys.getsizeof(np.float32(4))
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

    def fmag_all_update(self, f, addr, fmag, fmask, err_fmag, pbound=0.0):
        fdev = self.npy.fdev
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


class AuxiliaryWaveKernel(ab.AuxiliaryWaveKernel):

    def __init__(self, queue_thread=None):
        super(AuxiliaryWaveKernel, self).__init__()
        # and now initialise the cuda
        self.queue = queue_thread
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
                            np.float32(alpha),
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


class PoUpdateKernel(ab.PoUpdateKernel):

    def __init__(self, queue_thread=None):
        super(PoUpdateKernel, self).__init__()
        # and now initialise the cuda
        self.queue = queue_thread
        self.ob_update_cuda = load_kernel("ob_update")
        self.ob_update2_cuda = None # load_kernel("ob_update2")
        self.pr_update_cuda = load_kernel("pr_update")
        self.pr_update2_cuda = None

    def ob_update(self, addr, ob, obn, pr, ex, atomics=True):
        obsh = [np.int32(ax) for ax in ob.shape]
        prsh = [np.int32(ax) for ax in pr.shape]

        if atomics:
            num_pods = np.int32(addr.shape[0] * addr.shape[1])
            self.ob_update_cuda(ex, num_pods, prsh[1], prsh[2],
                                pr, prsh[0], prsh[1], prsh[2],
                                ob, obsh[0], obsh[1], obsh[2],
                                addr,
                                obn,
                                block=(32, 32, 1), grid=(int(num_pods), 1, 1), stream=self.queue)
        else:
            num_pods = np.int32(addr.shape[2] * addr.shape[3])
            if not self.ob_update2_cuda:
                self.ob_update2_cuda = load_kernel("ob_update2", {
                    "NUM_MODES": obsh[0],
                    "BDIM_X": 16,
                    "BDIM_Y": 16
                })

            #print('pods: {}'.format(num_pods))
            #print('address: {}'.format(addr.shape))
            # make a local stripped down clone of addr array for usage here:

            grid = [int(x/16) for x in ob.shape[-2:]]
            grid = (grid[0], grid[1], int(1))
            self.ob_update2_cuda(prsh[-1], obsh[0], num_pods, ob, obn, pr, ex, addr,
                                 block=(16,16, 1), grid=grid, stream=self.queue)

    def pr_update(self, addr, pr, prn, ob, ex, atomics=True):
        obsh = [np.int32(ax) for ax in ob.shape]
        prsh = [np.int32(ax) for ax in pr.shape]
        if atomics:
            num_pods = np.int32(addr.shape[0] * addr.shape[1])
            self.pr_update_cuda(ex, num_pods, prsh[1], prsh[2],
                                pr, prsh[0], prsh[1], prsh[2],
                                ob, obsh[0], obsh[1], obsh[2],
                                addr,
                                prn,
                                block=(32, 32, 1), grid=(int(num_pods), 1, 1), stream=self.queue)
        else:
            num_pods = np.int32(addr.shape[2] * addr.shape[3])
            if not self.pr_update2_cuda:
                self.pr_update2_cuda = load_kernel("pr_update2", {
                    "NUM_MODES": prsh[0],
                    "BDIM_X": 16,
                    "BDIM_Y": 16
                })
            grid = [int(x/16) for x in pr.shape[-2:]]
            grid = (grid[0], grid[1], int(1))
            self.pr_update2_cuda(prsh[-1], obsh[-2], obsh[-1],
                                 prsh[0], num_pods,
                                 pr, prn, ob, ex, addr,
                                 block=(16,16,1), grid=grid, stream=self.queue)


class PositionCorrectionKernel(ab.PositionCorrectionKernel):
    def __init__(self, aux, nmodes, queue_thread=None):
        super(PositionCorrectionKernel, self).__init__(aux, nmodes)
        # add kernels
        self.queue = queue_thread
        self._ob_shape = None
        self._ob_id = None
        self.fourier_error_cuda = load_kernel("fourier_error")
        self.error_reduce_cuda = load_kernel("error_reduce")
        self.build_aux_pc_cuda = load_kernel("build_aux_position_correction")


    def allocate(self):
        self.npy.fdev = gpuarray.zeros(self.fshape, dtype=np.float32)
        self.npy.ferr = gpuarray.zeros(self.fshape, dtype=np.float32)

    def build_aux(self, b_aux, addr, ob, pr):
        obr, obc = self._cache_object_shape(ob)
        self.build_aux_pc_cuda(b_aux,
                               pr,
                               np.int32(pr.shape[1]), np.int32(pr.shape[2]),
                               ob,
                               obr, obc,
                               addr,
                               block=(32, 32, 1), grid=(int(np.prod(addr.shape[:1])), 1, 1), stream=self.queue)

    def fourier_error(self, f, addr, fmag, fmask, mask_sum):
        fdev = self.npy.fdev
        ferr = self.npy.ferr
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
        # float_size = sys.getsizeof(np.float32(4))
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

    def update_addr_and_error_state(self, addr, error_state, mangled_addr, err_sum):
        '''
        updates the addresses and err state vector corresponding to the smallest error. I think this can be done on the cpu
        '''
        update_indices = err_sum < error_state
        log(4, "updating %s indices" % np.sum(update_indices))
        addr_cpu  = addr.get()
        addr_cpu[update_indices] = mangled_addr[update_indices]
        addr.set(addr_cpu)

        error_state[update_indices] = err_sum[update_indices]

    def _cache_object_shape(self, ob):
        oid = id(ob)

        if not oid == self._ob_id:
            self._ob_id = oid
            self._ob_shape = (np.int32(ob.shape[-2]), np.int32(ob.shape[-1]))

        return self._ob_shape