import numpy as np
from inspect import getfullargspec
from pycuda import gpuarray
from ptypy.utils.verbose import log
from . import load_kernel
from ..array_based import kernels as ab
from ..array_based.base import Adict

class FourierUpdateKernel(ab.FourierUpdateKernel):

    def __init__(self, aux, nmodes=1, queue_thread=None):
        super(FourierUpdateKernel, self).__init__(aux,  nmodes=nmodes)
        self.queue = queue_thread
        self.fmag_all_update_cuda = load_kernel("fmag_all_update")
        self.fourier_error_cuda = load_kernel("fourier_error")
        self.fourier_error2_cuda = None 
        self.error_reduce_cuda = load_kernel("error_reduce")
        self.fourier_update_cuda = None

    def allocate(self):
        self.npy.fdev = gpuarray.zeros(self.fshape, dtype=np.float32)
        self.npy.ferr = gpuarray.zeros(self.fshape, dtype=np.float32)

    def fourier_error(self, f, addr, fmag, fmask, mask_sum):
        fdev = self.npy.fdev
        ferr = self.npy.ferr
        if True:
            # version going over all modes in a single thread (faster)
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
        else:
            # version using one thread per mode + shared mem reduction (slower)
            if self.fourier_error2_cuda is None:
                self.fourier_error2_cuda = load_kernel("fourier_error2")
            bx = 16
            by = 16
            bz = int(self.nmodes)
            blk = (bx, by, bz)
            grd = (int((self.fshape[2] + bx-1) // bx), 
                                        int((self.fshape[1] + by-1) // by),
                                        int(self.fshape[0]))
            #print('block={}, grid={}, fshape={}'.format(blk, grd, self.fshape))
            self.fourier_error2_cuda(np.int32(self.nmodes),
                                    f,
                                    fmask,
                                    fmag,
                                    fdev,
                                    ferr,
                                    mask_sum,
                                    addr,
                                    np.int32(self.fshape[1]),
                                    np.int32(self.fshape[2]),
                                    block=blk,
                                    grid=grd,
                                    shared=int(bx*by*bz*4),
                                    stream=self.queue)

    def error_reduce(self, addr, err_fmag):
        # import sys
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

    # Note: this was a test to join the kernels, but it's > 2x slower!
    def fourier_update(self, f, addr, fmag, fmask, mask_sum, err_fmag, pbound=0):
        if self.fourier_update_cuda  is None:
            self.fourier_update_cuda = load_kernel("fourier_update")
        fdev = self.npy.fdev
        ferr = self.npy.ferr

        bx = 16
        by = 16
        bz = int(self.nmodes)
        blk = (bx, by, bz)
        grd = (int((self.fshape[2] + bx-1) // bx), 
                int((self.fshape[1] + by-1) // by),
                int(self.fshape[0]))
        smem = int(bx*by*bz*4)
        self.fourier_update_cuda(np.int32(self.nmodes),
                                    f,
                                    fmask,
                                    fmag,
                                    fdev,
                                    ferr,
                                    mask_sum,
                                    addr,
                                    err_fmag,
                                    np.float32(pbound),
                                    np.int32(self.fshape[1]),
                                    np.int32(self.fshape[2]),
                                    block=blk,
                                    grid=grd,
                                    shared=smem,
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


class GradientDescentKernel(ab.GradientDescentKernel):

    def __init__(self, aux, nmodes=1, queue=None):
        super().__init__(aux, nmodes)
        self.queue = queue
        
        self.gpu = Adict()
        self.gpu.LLden = None
        self.gpu.LLerr = None 
        self.gpu.Imodel = None 

        subs = {
            'CTYPE': 'complex<float>' if self.ctype == np.complex64 else 'complex<double>',
            'FTYPE': 'float' if self.ftype == np.float32 else 'double'
        }
        self.make_model_cuda = load_kernel('make_model', subs)

    def allocate(self):
        self.gpu.LLden = gpuarray.zeros(self.fshape, dtype=self.ftype)
        self.gpu.LLerr = gpuarray.zeros(self.fshape, dtype=self.ftype)
        self.gpu.Imodel = gpuarray.zeros(self.fshape, dtype=self.ftype)

    def make_model(self, b_aux):
        # reference shape
        sh = self.fshape

        # batch buffers
        Imodel = self.gpu.Imodel
        aux = b_aux

        # dimensions / grid
        z = np.int32(sh[0])
        y = np.int32(self.nmodes)
        x = np.int32(sh[1] * sh[2])
        bx = 1024
        self.make_model_cuda(aux, Imodel, z, y, x,
                             block=(bx, 1, 1),
                             grid=(int((x + bx - 1) // bx), 1, int(z)))

    def make_a012(self, b_f, b_a, b_b, I):
        pass

    def fill_b(self, Brenorm, w, B):
        pass

    def error_reduce(self, err_sum):
        pass

    def main(self, b_aux, w, I):
        pass

class PoUpdateKernel(ab.PoUpdateKernel):

    def __init__(self, queue_thread=None, denom_type=np.complex64):
        super(PoUpdateKernel, self).__init__()
        # and now initialise the cuda
        if denom_type == np.complex64:
            dtype = 'complex<float>'
        elif denom_type == np.float32:
            dtype = 'float'
        else:
            raise ValueError('only complex64 and float32 types supported')
        self.dtype = dtype
        self.queue = queue_thread
        self.ob_update_cuda = load_kernel("ob_update", {
            'DENOM_TYPE': dtype
        })
        self.ob_update2_cuda = None # load_kernel("ob_update2")
        self.pr_update_cuda = load_kernel("pr_update", {
            'DENOM_TYPE': dtype
        })
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
                    "BDIM_Y": 16,
                    'DENOM_TYPE': self.dtype
                })

            # print('pods: {}'.format(num_pods))
            # print('address: {}'.format(addr.shape))
            # print('ob: {}'.format(ob.shape))
            # print('obn: {}'.format(obn.shape))
            # print('ex: {}'.format(ex.shape))
            # print('prsh: {}'.format(prsh))
            # make a local stripped down clone of addr array for usage here:

            grid = [int((x+15)//16) for x in ob.shape[-2:]]
            grid = (grid[0], grid[1], int(1))
            self.ob_update2_cuda(prsh[-1], obsh[0], num_pods, obsh[-2], 
                                 prsh[0], 
                                 np.int32(ex.shape[0]), 
                                 np.int32(ex.shape[1]), 
                                 np.int32(ex.shape[2]), 
                                 ob, obn, pr, ex, addr,
                                 block=(16,16, 1), grid=grid, stream=self.queue)

    def pr_update(self, addr, pr, prn, ob, ex, atomics=True):
        obsh = [np.int32(ax) for ax in ob.shape]
        prsh = [np.int32(ax) for ax in pr.shape]
        #print('Ob sh: {}, pr sh: {}'.format(obsh, prsh))
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
                    "BDIM_Y": 16,
                    'DENOM_TYPE': self.dtype
                })

            # print('pods: {}'.format(num_pods))
            # print('address: {}'.format(addr.shape))
            # print('ex: {}'.format(ex.shape))
            # print('prsh: {}'.format(prsh))
            # print('ob: {}'.format(ob.shape))

            grid = [int((x+15)//16) for x in pr.shape[-2:]]
            grid = (grid[0], grid[1], int(1))
            self.pr_update2_cuda(prsh[-1], obsh[-2], obsh[-1],
                                 prsh[0], obsh[0], num_pods,
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

class DerivativesKernel:
    def __init__(self, dtype, stream=None):
        if dtype == np.float32:
            stype = "float"
        elif dtype == np.complex64:
            stype = "complex<float>"
        else:
            raise NotImplementedError("delxf is only implemented for float32 and complex64")
        
        self.queue = stream
        self.dtype = dtype
        self.last_axis_block = (256, 4, 1)
        self.mid_axis_block = (256, 4, 1)
        
        self.delxf_last = load_kernel("delx_last", file="delx_last.cu", subs={
            'IS_FORWARD': 'true',
            'BDIM_X': str(self.last_axis_block[0]),
            'BDIM_Y': str(self.last_axis_block[1]),
            'DTYPE': stype
        })
        self.delxb_last = load_kernel("delx_last", file="delx_last.cu", subs={
            'IS_FORWARD': 'false',
            'BDIM_X': str(self.last_axis_block[0]),
            'BDIM_Y': str(self.last_axis_block[1]),
            'DTYPE': stype
        })
        self.delxf_mid = load_kernel("delx_mid", file="delx_mid.cu", subs={
            'IS_FORWARD': 'true',
            'BDIM_X': str(self.mid_axis_block[0]),
            'BDIM_Y': str(self.mid_axis_block[1]),
            'DTYPE': stype
        })
        self.delxb_mid = load_kernel("delx_mid", file="delx_mid.cu", subs={
            'IS_FORWARD': 'false',
            'BDIM_X': str(self.mid_axis_block[0]),
            'BDIM_Y': str(self.mid_axis_block[1]),
            'DTYPE': stype
        })

    def delxf(self, input, out, axis=-1):
        if input.dtype != self.dtype:
            raise ValueError('Invalid input data type')

        if axis < 0:
            axis = input.ndim + axis
        axis = np.int32(axis)

        if axis == input.ndim - 1:
            flat_dim = np.int32(np.product(input.shape[0:-1]))
            self.delxf_last(input, out, flat_dim, np.int32(input.shape[axis]), 
                block=self.last_axis_block, 
                grid=(
                    int((flat_dim + self.last_axis_block[1] - 1) // self.last_axis_block[1]), 
                    1, 1),
                stream=self.queue
            )
        else:
            lower_dim = np.int32(np.product(input.shape[(axis+1):]))
            higher_dim = np.int32(np.product(input.shape[:axis]))
            gx = int((lower_dim + self.mid_axis_block[0] - 1) // self.mid_axis_block[0])
            gy = 1
            gz = int(higher_dim)
            self.delxf_mid(input, out, lower_dim, higher_dim, np.int32(input.shape[axis]),
                block=self.mid_axis_block,
                grid=(gx, gy, gz), 
                stream=self.queue
            )


    def delxb(self, input, out, axis=-1):
        if input.dtype != self.dtype:
            raise ValueError('Invalid input data type')

        if axis < 0:
            axis = input.ndim + axis
        axis = np.int32(axis)
        
        if axis == input.ndim - 1:
            flat_dim = np.int32(np.product(input.shape[0:-1]))
            self.delxb_last(input, out, flat_dim, np.int32(input.shape[axis]), 
                block=self.last_axis_block, 
                grid=(
                    int((flat_dim + self.last_axis_block[1] - 1) // self.last_axis_block[1]),
                    1, 1),
                stream=self.queue
            )
        else:
            lower_dim = np.int32(np.product(input.shape[(axis+1):]))
            higher_dim = np.int32(np.product(input.shape[:axis]))
            gx = int((lower_dim + self.mid_axis_block[0] - 1) // self.mid_axis_block[0])
            gy = 1
            gz = int(higher_dim)
            self.delxb_mid(input, out, lower_dim, higher_dim, np.int32(input.shape[axis]),
                block=self.mid_axis_block,
                grid=(gx, gy, gz),
                stream=self.queue
            )
        