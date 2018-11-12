'''
This maps bjoerns kernels in accelerate.ocl.np_kernels to the ones in array_based

'''

import numpy as np
from collections import OrderedDict
from error_metrics import far_field_error
import object_probe_interaction as opi
import constraints as con

class Adict(object):

    def __init__(self):
        pass


class BaseKernel(object):

    def __init__(self):
        self.verbose = False
        self.npy = Adict()
        self.benchmark = OrderedDict()

    def log(self, x):
        if self.verbose:
            print(x)


class Fourier_update_kernel(BaseKernel):

    def __init__(self, pbound=0.0):
        self.pbound = np.float32(pbound)

    def test(self, I, mask, f):
        """
        Test arrays for shape and data type
        """
        assert I.dtype == np.float32
        assert I.shape == self.fshape
        assert mask.dtype == np.float32
        assert mask.shape == self.fshape
        assert f.dtype == np.complex64
        assert f.shape == self.ishape

    def allocate(self, shape, nmodes=1):
        """
        Allocate memory according to the number of modes and
        shape of the diffraction stack.
        """
        assert len(shape) == 2
        self.nmodes = np.int32(nmodes)
        self.fshape = shape
        self.ishape = (self.nmodes * shape[0], shape[1], shape[2])

        self.framesize = np.int32(np.prod(shape[-2:]))

        # temporary buffer arrays
        self.npy.fdev = np.zeros(shape, dtype=np.float32)
        self.npy.ferr = np.zeros(shape, dtype=np.float32)

        self.kernels = [
            'fourier_error',
            'error_reduce',
            'fmag_all_update'
        ]

    def npy_fourier_error(self, f, fmag, fdev, ferr, fmask, mask_sum):
        ferr[:] = far_field_error(f, fmag, fmask)
        fdev[:] = af - fmag # needed?

    def npy_error_reduce(self, ferr, err_fmag):
        return

    def _npy_calc_fm(self,fm, fmask, fmag, fdev, err_fmag):
        return

    def _npy_fmag_update(self, f, fm):
        return

    def npy_fmag_all_update(self, f, fmask, fmag, fdev, err_fmag):
        return

class Auxiliary_wave_kernel(BaseKernel):

    def __init__(self, queue_thread=None):

        super(Auxiliary_wave_kernel, self).__init__(queue_thread)

    def configure(self, ob, addr, alpha=1.0):

        self.batch_offset = 0
        self.alpha = np.float32(alpha)
        self.ob_shape = (np.int32(ob.shape[-2]), np.int32(ob.shape[-1]))

        self.nviews, self.nmodes, self.ncoords, self.naxes = addr.shape
        self.ocl_wg_size = (1, 1, 32)

    @property
    def batch_offset(self):
        return self._offset

    @batch_offset.setter
    def batch_offset(self, x):
        self._offset = np.int32(x)

    def npy_build_aux(self, aux, ob, pr, ex, addr):

        probe_and_object = opi.scan_and_multiply(pr, ob, ex.shape, addr)
        aux[:] = opi.difference_map_realspace_constraint(probe_and_object, ex, self.alpha)

    def npy_build_exit(self, aux, ob, pr, ex, addr):
        pbound = None
        err_fmag = np.zeros((self.nviews/self.nmodes))
        probe_object = opi.scan_and_multiply(pr, ob, ex.shape, addr)
        df = con.get_difference(addr, self.alpha, aux, err_fmag, ex, pbound, probe_object)
        ex += df
        aux[:] = df  # so should make get_difference in-place in aux


class PO_update_kernel(BaseKernel):

    def __init__(self, queue_thread=None):

        super(PO_update_kernel, self).__init__(queue_thread)

    def configure(self, ob, pr, addr):

        self.batch_offset = 0
        self.ob_shape = tuple([np.int32(ax) for ax in ob.shape])
        self.pr_shape = tuple([np.int32(ax) for ax in pr.shape])
        # self.ob_shape = (np.int32(ob.shape[-2]),np.int32(ob.shape[-1]))
        # self.pr_shape = (np.int32(pr.shape[-2]),np.int32(pr.shape[-1]))

        self.nviews, self.nmodes, self.ncoords, self.naxes = addr.shape
        self.num_pods = np.int32(self.nviews * self.nmodes)
        self.ocl_wg_size = (16, 16)

    @property
    def batch_offset(self):
        return self._offset

    @batch_offset.setter
    def batch_offset(self, x):
        self._offset = np.int32(x)

    def npy_ob_update(self, ob, obn, pr, ex, addr):
        obsh = self.ob_shape
        prsh = self.pr_shape
        sh = addr.shape
        flat_addr = addr.reshape(sh[0] * sh[1], sh[2], sh[3])
        rows, cols = ex.shape[-2:]
        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] += \
                pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols].conj() * \
                ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols]

            obn[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] += \
                pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols].conj() * \
                pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols]
        return

    def npy_pr_update(self, pr, prn, ob, ex, addr):
        obsh = self.ob_shape
        prsh = self.pr_shape
        sh = addr.shape
        flat_addr = addr.reshape(sh[0] * sh[1], sh[2], sh[3])
        rows, cols = ex.shape[-2:]
        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols] += \
                ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols].conj() * \
                ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols]
            prn[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols] += \
                ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols].conj() * \
                ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols]
        return










