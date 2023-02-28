import numpy as np
from inspect import getfullargspec
from pycuda import gpuarray
from ptypy.utils.verbose import log, logger
from . import load_kernel
from .array_utils import CropPadKernel
from .array_utils import MaxAbs2Kernel
from ..base import kernels as ab
from ..base.kernels import Adict

def choose_fft(fft_type, arr_shape):
    dims_are_powers_of_two = True
    rows = arr_shape[0]
    columns = arr_shape[1]
    if rows != columns or rows not in [16, 32, 64, 128, 256, 512, 1024, 2048]:
        dims_are_powers_of_two = False
    if fft_type=='cuda' and not dims_are_powers_of_two:
        logger.warning('cufft: array dimensions are not powers of two (16 to 2048) - using Reikna instead')
        from ptypy.accelerate.cuda_pycuda.fft import FFT
    elif fft_type=='cuda' and dims_are_powers_of_two:
        try:
            from ptypy.accelerate.cuda_pycuda.cufft import FFT_cuda as FFT
        except:
            import filtered_cufft
            logger.warning('Unable to import cufft version - using Reikna instead')
            from ptypy.accelerate.cuda_pycuda.fft import FFT
    elif fft_type=='skcuda':
        try:
            from ptypy.accelerate.cuda_pycuda.cufft import FFT_skcuda as FFT
        except:
            logger.warning('Unable to import skcuda.fft version - using Reikna instead')
            from ptypy.accelerate.cuda_pycuda.fft import FFT
    else:
        from ptypy.accelerate.cuda_pycuda.fft import FFT
    return FFT

class PropagationKernel:

    def __init__(self, aux, propagator, queue_thread=None, fft='reikna'):
        self.aux = aux
        self._queue = queue_thread
        self.prop_type = propagator.p.propagation
        self.fw = None
        self.bw = None
        self._fft1 = None
        self._fft2 = None
        self._p = propagator
        self._fft_type = fft

    def allocate(self):

        aux = self.aux
        FFT = choose_fft(self._fft_type, aux.shape[-2:])

        if self.prop_type == 'farfield':

            self._do_crop_pad = (self._p.crop_pad != 0).any()
            if self._do_crop_pad:
                aux_shape = tuple(np.array(aux.shape) + np.append([0],self._p.crop_pad))
                self._tmp = np.zeros(aux_shape, dtype=aux.dtype)
                self._CPK = CropPadKernel(queue=self._queue)
            else:
                self._tmp = aux

            self._fft1 = FFT(self._tmp, self.queue,
                             pre_fft=self._p.pre_fft,
                             post_fft=self._p.post_fft,
                             symmetric=True,
                             forward=True)
            self._fft2 = FFT(self._tmp, self.queue,
                             pre_fft=self._p.pre_ifft,
                             post_fft=self._p.post_ifft,
                             symmetric=True,
                             forward=False)
            if self._do_crop_pad:
                self._tmp = gpuarray.to_gpu(self._tmp)

            def _fw(x,y):
                if self._do_crop_pad:
                    self._CPK.crop_pad_2d_simple(self._tmp, x)
                    self._fft1.ft(self._tmp, self._tmp)
                    self._CPK.crop_pad_2d_simple(y, self._tmp)
                else:
                    self._fft1.ft(x,y)

            def _bw(x,y):
                if self._do_crop_pad:
                    self._CPK.crop_pad_2d_simple(self._tmp, x)
                    self._fft2.ift(self._tmp, self._tmp)
                    self._CPK.crop_pad_2d_simple(y, self._tmp)
                else:
                    self._fft2.ift(x,y)
            
            self.fw = _fw
            self.bw = _bw

        elif self.prop_type == "nearfield":
            self._fft1 = FFT(aux, self.queue,
                             post_fft=self._p.kernel,
                             symmetric=True,
                             forward=True)
            self._fft2 = FFT(aux, self.queue,
                             post_fft=self._p.ikernel,
                             inplace=True,
                             symmetric=True,
                             forward=True)
            self._fft3 = FFT(aux, self.queue,
                             symmetric=True,
                             forward=False)

            def _fw(x,y):
                self._fft1.ft(x,y)
                self._fft3.ift(y,y)
            
            def _bw(x,y):
                self._fft2.ft(x,y)
                self._fft3.ift(y,y)
                
            self.fw = _fw
            self.bw = _bw
        else:
            logger.warning("Unable to select propagator %s, only nearfield and farfield are supported" %self.prop_type)

    @property
    def queue(self):
        return self._queue

    @queue.setter
    def queue(self, queue):
        self._queue = queue
        self._fft1.queue = queue
        self._fft2.queue = queue
        if self.prop_type == "nearfield":
            self._fft3.queue = queue

class FourierSupportKernel:
    def __init__(self, support, queue_thread=None, fft='reikna'):
        self.support = support
        self.queue = queue_thread
        self._fft_type = fft
    def allocate(self):
        FFT = choose_fft(self._fft_type, self.support.shape[-2:])

        self._fft1 = FFT(self.support, self.queue,
                        post_fft=self.support,
                        symmetric=True,
                        forward=True)
        self._fft2 = FFT(self.support, self.queue,
                        symmetric=True,
                        forward=False)
    def apply_fourier_support(self,x):
        self._fft1.ft(x,x)
        self._fft2.ift(x,x)

class RealSupportKernel:
    def __init__(self, support):
        self.support = support
    def allocate(self):
        self.support = gpuarray.to_gpu(self.support)
    def apply_real_support(self, x):
        x *= self.support

class FourierUpdateKernel(ab.FourierUpdateKernel):

    def __init__(self, aux, nmodes=1, queue_thread=None, accumulate_type='float', math_type='float'):
        super(FourierUpdateKernel, self).__init__(aux,  nmodes=nmodes)

        if accumulate_type not in ['float', 'double']:
            raise ValueError('Only float or double types are supported')
        if math_type not in ['float', 'double']:
            raise ValueError('Only float or double types are supported')
        self.accumulate_type = accumulate_type
        self.math_type = math_type
        self.queue = queue_thread
        self.fmag_all_update_cuda = load_kernel("fmag_all_update", {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'MATH_TYPE': self.math_type
        })
        self.fmag_update_nopbound_cuda = None
        self.fourier_deviation_cuda = None
        self.fourier_error_cuda = load_kernel("fourier_error", {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'MATH_TYPE': self.math_type
        })
        self.fourier_error2_cuda = None
        self.error_reduce_cuda = load_kernel("error_reduce", {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'ACC_TYPE': self.accumulate_type,
            'BDIM_X': 32,
            'BDIM_Y': 32,
        })
        self.fourier_update_cuda = None
        self.log_likelihood_cuda, self.log_likelihood2_cuda = load_kernel(
            ("log_likelihood", "log_likelihood2"), {
                'IN_TYPE': 'float',
                'OUT_TYPE': 'float',
                'MATH_TYPE': self.math_type
            },
            "log_likelihood.cu")
        self.exit_error_cuda = load_kernel("exit_error", {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'MATH_TYPE': self.math_type
        })

        self.gpu = Adict()
        self.gpu.fdev = None
        self.gpu.ferr = None

    def allocate(self):
        self.gpu.fdev = gpuarray.zeros(self.fshape, dtype=np.float32)
        self.gpu.ferr = gpuarray.zeros(self.fshape, dtype=np.float32)

    def fourier_error(self, f, addr, fmag, fmask, mask_sum):
        fdev = self.gpu.fdev
        ferr = self.gpu.ferr
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

    def fourier_deviation(self, f, addr, fmag):
        fdev = self.gpu.fdev
        if self.fourier_deviation_cuda is None:
            self.fourier_deviation_cuda = load_kernel("fourier_deviation",{
                'IN_TYPE': 'float',
                'OUT_TYPE': 'float',
                'MATH_TYPE': self.math_type
            })
        bx = 64
        by = 1
        self.fourier_deviation_cuda(np.int32(self.nmodes),
                                f,
                                fmag,
                                fdev,
                                addr,
                                np.int32(self.fshape[1]),
                                np.int32(self.fshape[2]),
                                block=(bx, by, 1),
                                grid=(1, int((self.fshape[2] + by - 1)//by), int(fmag.shape[0])),
                                stream=self.queue)


    def error_reduce(self, addr, err_sum):
        self.error_reduce_cuda(self.gpu.ferr,
                               err_sum,
                               np.int32(self.fshape[1]),
                               np.int32(self.fshape[2]),
                               block=(32, 32, 1),
                               grid=(int(err_sum.shape[0]), 1, 1),
                               stream=self.queue)

    def fmag_all_update(self, f, addr, fmag, fmask, err_fmag, pbound=0.0):
        fdev = self.gpu.fdev
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
    
    def fmag_update_nopbound(self, f, addr, fmag, fmask):
        fdev = self.gpu.fdev
        bx = 64
        by = 1
        if self.fmag_update_nopbound_cuda is None:
            self.fmag_update_nopbound_cuda = load_kernel("fmag_update_nopbound", {
                'IN_TYPE': 'float',
                'OUT_TYPE': 'float',
                'MATH_TYPE': self.math_type
            })
        self.fmag_update_nopbound_cuda(f,
                                  fmask,
                                  fmag,
                                  fdev,
                                  addr,
                                  np.int32(self.fshape[1]),
                                  np.int32(self.fshape[2]),
                                  block=(bx, by, 1),
                                  grid=(1, 
                                    int((self.fshape[2] + by - 1) // by), 
                                    int(fmag.shape[0]*self.nmodes)),
                                  stream=self.queue)

    # Note: this was a test to join the kernels, but it's > 2x slower!
    def fourier_update(self, f, addr, fmag, fmask, mask_sum, err_fmag, pbound=0):
        if self.fourier_update_cuda is None:
            self.fourier_update_cuda = load_kernel("fourier_update")
        fdev = self.gpu.fdev
        ferr = self.gpu.ferr

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

    def log_likelihood(self, b_aux, addr, mag, mask, err_phot):
        ferr = self.gpu.ferr
        self.log_likelihood_cuda(np.int32(self.nmodes),
                                 b_aux,
                                 mask,
                                 mag,
                                 addr,
                                 ferr,
                                 np.int32(self.fshape[1]),
                                 np.int32(self.fshape[2]),
                                 block=(32, 32, 1),
                                 grid=(int(mag.shape[0]), 1, 1),
                                 stream=self.queue)
        # TODO: we might want to move this call outside of here
        self.error_reduce(addr, err_phot)

    def log_likelihood2(self, b_aux, addr, mag, mask, err_phot):
        ferr = self.gpu.ferr
        bx = 64
        by = 1
        self.log_likelihood2_cuda(np.int32(self.nmodes),
                                 b_aux,
                                 mask,
                                 mag,
                                 addr,
                                 ferr,
                                 np.int32(self.fshape[1]),
                                 np.int32(self.fshape[2]),
                                 block=(bx, by, 1),
                                 grid=(1, int((self.fshape[1] + by - 1) // by), int(mag.shape[0])),
                                 stream=self.queue)
        # TODO: we might want to move this call outside of here
        self.error_reduce(addr, err_phot)

    def exit_error(self, aux, addr):
        sh = addr.shape
        maxz = sh[0]
        ferr = self.gpu.ferr
        self.exit_error_cuda(np.int32(self.nmodes),
                             aux,
                             ferr,
                             addr,
                             np.int32(self.fshape[1]),
                             np.int32(self.fshape[2]),
                             block=(32, 32, 1),
                             grid=(int(maxz), 1, 1),
                             stream=self.queue)

    # DEPRECATED?
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

    def __init__(self, queue_thread=None, math_type = 'float'):
        super(AuxiliaryWaveKernel, self).__init__()
        # and now initialise the cuda
        self.queue = queue_thread
        self._ob_shape = None
        self._ob_id = None
        self.math_type = math_type
        if math_type not in ['float', 'double']:
            raise ValueError('Only double or float math is supported')
        self.make_aux_cuda, self.make_aux2_cuda = load_kernel(
            ("make_aux", "make_aux2"), {
                'IN_TYPE': 'float',
                'OUT_TYPE': 'float',
                'MATH_TYPE': self.math_type
            }, "make_aux.cu")
        self.make_exit_cuda = load_kernel("make_exit", {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'MATH_TYPE': self.math_type
        })
        self.build_aux_no_ex_cuda, self.build_aux2_no_ex_cuda = load_kernel(
            ("build_aux_no_ex", "build_aux2_no_ex"), {
                'IN_TYPE': 'float',
                'OUT_TYPE': 'float',
                'MATH_TYPE': self.math_type
            }, "build_aux_no_ex.cu")
        # self.build_exit_alpha_tau_cuda = load_kernel("build_exit_alpha_tau", {
        #     'IN_TYPE': 'float',
        #     'OUT_TYPE': 'float',
        #     'MATH_TYPE': self.math_type
        # })

    # DEPRECATED?
    def load(self, aux, ob, pr, ex, addr):
        super(AuxiliaryWaveKernel, self).load(aux, ob, pr, ex, addr)
        for key, array in self.npy.__dict__.items():
            self.ocl.__dict__[key] = gpuarray.to_gpu(array)

    def make_aux(self, b_aux, addr, ob, pr, ex, c_po=1.0, c_e=0.0):
        obr, obc = self._cache_object_shape(ob)
        sh = addr.shape
        nmodes = sh[1]
        maxz = sh[0]
        self.make_aux_cuda(b_aux,
                            ex,
                            np.int32(ex.shape[1]), np.int32(ex.shape[2]),
                            pr,
                            np.int32(ex.shape[1]), np.int32(ex.shape[2]),
                            ob,
                            obr, obc,
                            addr,
                            np.float32(c_po) if ex.dtype == np.complex64 else np.float64(c_po),
                            np.float32(c_e) if ex.dtype == np.complex64 else np.float64(c_e),
                            block=(32, 32, 1), grid=(int(maxz * nmodes), 1, 1), stream=self.queue)

    def make_aux2(self, b_aux, addr, ob, pr, ex, c_po=1.0, c_e=0.0):
        obr, obc = self._cache_object_shape(ob)
        sh = addr.shape
        nmodes = sh[1]
        maxz = sh[0]
        bx = 64
        by = 1
        self.make_aux2_cuda(b_aux,
                            ex,
                            np.int32(ex.shape[1]), np.int32(ex.shape[2]),
                            pr,
                            np.int32(ex.shape[1]), np.int32(ex.shape[2]),
                            ob,
                            obr, obc,
                            addr,
                            np.float32(c_po) if ex.dtype == np.complex64 else np.float64(c_po),
                            np.float32(c_e) if ex.dtype == np.complex64 else np.float64(c_e),
                            block=(bx, by, 1),
                            grid=(
                                1, 
                                int((ex.shape[1] + by - 1)//by), 
                                int(maxz * nmodes)), 
                            stream=self.queue)


    def make_exit(self, b_aux, addr, ob, pr, ex, c_a=1.0, c_po=0.0, c_e=-1.0):
        obr, obc = self._cache_object_shape(ob)
        sh = addr.shape
        nmodes = sh[1]
        maxz = sh[0]
        self.make_exit_cuda(b_aux,
                             ex,
                             np.int32(ex.shape[1]), np.int32(ex.shape[2]),
                             pr,
                             np.int32(ex.shape[1]), np.int32(ex.shape[2]),
                             ob,
                             obr, obc,
                             addr,
                             np.float32(c_a) if ex.dtype == np.complex64 else np.float64(c_a),
                             np.float32(c_po) if ex.dtype == np.complex64 else np.float64(c_po),
                             np.float32(c_e) if ex.dtype == np.complex64 else np.float64(c_e),
                             block=(32, 32, 1), grid=(int(maxz * nmodes), 1, 1), stream=self.queue)

    def build_aux2(self, b_aux, addr, ob, pr, ex, alpha=1.0):
        # DM only, legacy. also make_aux2 does no exit in the parent
        self.make_aux2(b_aux, addr, ob, pr, ex, 1.+alpha, -alpha)

    """
    def build_exit_alpha_tau(self, b_aux, addr, ob, pr, ex, alpha=1, tau=1):
        obr, obc = self._cache_object_shape(ob)
        sh = addr.shape
        nmodes = sh[1]
        maxz = sh[0]
        bx = 64
        by = 1
        self.build_exit_alpha_tau_cuda(b_aux,
                                       ex,
                                       np.int32(ex.shape[1]), np.int32(ex.shape[2]),
                                       pr,
                                       np.int32(ex.shape[1]), np.int32(ex.shape[2]),
                                       ob,
                                       obr, obc,
                                       addr,
                                       np.float32(alpha), np.float32(tau),
                                       block=(bx, by, 1), 
                                       grid=(1, int((ex.shape[1] + by - 1) // by), int(maxz * nmodes)), 
                                       stream=self.queue)
    """
    def build_aux_no_ex(self, b_aux, addr, ob, pr, fac=1.0, add=False):
        obr, obc = self._cache_object_shape(ob)
        sh = addr.shape
        nmodes = sh[1]
        maxz = sh[0]
        self.build_aux_no_ex_cuda(b_aux,
                                  np.int32(b_aux.shape[-2]),
                                  np.int32(b_aux.shape[-1]),
                                  pr,
                                  np.int32(pr.shape[-2]),
                                  np.int32(pr.shape[-1]),
                                  ob,
                                  obr, obc,
                                  addr,
                                  np.float32(fac) if pr.dtype == np.complex64 else np.float64(fac),
                                  np.int32(add),
                                  block=(32, 32, 1),
                                  grid=(int(maxz * nmodes), 1, 1),
                                  stream=self.queue)


    def build_aux2_no_ex(self, b_aux, addr, ob, pr, fac=1.0, add=False):
        obr, obc = self._cache_object_shape(ob)
        sh = addr.shape
        nmodes = sh[1]
        maxz = sh[0]
        bx = 64
        by = 1
        self.build_aux2_no_ex_cuda(b_aux,
                                  np.int32(b_aux.shape[-2]),
                                  np.int32(b_aux.shape[-1]),
                                  pr,
                                  np.int32(pr.shape[-2]),
                                  np.int32(pr.shape[-1]),
                                  ob,
                                  obr, obc,
                                  addr,
                                  np.float32(fac) if pr.dtype == np.complex64 else np.float64(fac),
                                  np.int32(add),
                                  block=(bx, by, 1),
                                  grid=(1, int((b_aux.shape[-2] + by - 1)//by), int(maxz * nmodes)),
                                  stream=self.queue)
    
    
    def _cache_object_shape(self, ob):
        oid = id(ob)

        if not oid == self._ob_id:
            self._ob_id = oid
            self._ob_shape = (np.int32(ob.shape[-2]), np.int32(ob.shape[-1]))

        return self._ob_shape


class GradientDescentKernel(ab.GradientDescentKernel):

    def __init__(self, aux, nmodes=1, queue=None, accumulate_type = 'double', math_type='float'):
        super().__init__(aux, nmodes)
        self.queue = queue
        self.accumulate_type = accumulate_type
        self.math_type = math_type
        if (accumulate_type not in ['double', 'float']) or (math_type not in ['double', 'float']):
            raise ValueError("accumulate and math types must be double for float")
 
        self.gpu = Adict()
        self.gpu.LLden = None
        self.gpu.LLerr = None
        self.gpu.Imodel = None

        subs = {
            'IN_TYPE': 'float' if self.ftype == np.float32 else 'double',
            'OUT_TYPE': 'float' if self.ftype == np.float32 else 'double',
            'ACC_TYPE': self.accumulate_type,
            'MATH_TYPE': self.math_type
        }
        self.make_model_cuda = load_kernel('make_model', subs)
        self.make_a012_cuda = load_kernel('make_a012', subs)
        self.error_reduce_cuda = load_kernel('error_reduce', {
            **subs,
            'OUT_TYPE': 'float' if self.ftype == np.float32 else 'double',
            'BDIM_X': 32,
            'BDIM_Y': 32
        })
        self.fill_b_cuda, self.fill_b_reduce_cuda = load_kernel(
            ('fill_b', 'fill_b_reduce'), 
            {
                **subs, 
                'BDIM_X': 1024,
                'OUT_TYPE': 'float' if self.ftype == np.float32 else 'double'
            },
            file="fill_b.cu")
        self.main_cuda = load_kernel('gd_main', subs)
        self.floating_intensity_cuda_step1, self.floating_intensity_cuda_step2 = \
            load_kernel(('step1', 'step2'), subs,'intens_renorm.cu')

    def allocate(self):
        self.gpu.LLden = gpuarray.zeros(self.fshape, dtype=self.ftype)
        self.gpu.LLerr = gpuarray.zeros(self.fshape, dtype=self.ftype)
        self.gpu.Imodel = gpuarray.zeros(self.fshape, dtype=self.ftype)
        tmp = np.ones((self.fshape[0],), dtype=self.ftype)
        self.gpu.fic_tmp = gpuarray.to_gpu(tmp)

        # temporary array for the reduction in fill_b
        sh = (3, int((np.prod(self.fshape)*self.nmodes + 1023) // 1024))
        self.gpu.Btmp = gpuarray.zeros(sh, dtype=np.float64 if self.accumulate_type == 'double' else np.float32)

    def make_model(self, b_aux, addr):
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
                             grid=(int((x + bx - 1) // bx), 1, int(z)),
                             stream=self.queue)

    def make_a012(self, b_f, b_a, b_b, addr, I, fic):
        # reference shape (= GPU global dims)
        sh = I.shape

        # stopper
        maxz = I.shape[0]

        A0 = self.gpu.Imodel
        A1 = self.gpu.LLerr
        A2 = self.gpu.LLden

        z = np.int32(sh[0])
        maxz = np.int32(maxz)
        y = np.int32(self.nmodes)
        x = np.int32(sh[1]*sh[2])
        bx = 1024
        self.make_a012_cuda(b_f, b_a, b_b, I, fic,
                            A0, A1, A2, z, y, x, maxz,
                            block=(bx, 1, 1),
                            grid=(int((x + bx - 1) // bx), 1, int(z)),
                            stream=self.queue)

    def fill_b(self, addr, Brenorm, w, B):
        # stopper
        maxz = w.shape[0]

        A0 = self.gpu.Imodel
        A1 = self.gpu.LLerr
        A2 = self.gpu.LLden

        sz = np.int32(np.prod(w.shape))
        blks = int((sz + 1023) // 1024)
        # print('blocks={}, Btmp={}, fshape={}, wshape={}, modes={}'.format(blks, self.gpu.Btmp.shape, self.fshape, w.shape, self.nmodes))
        assert self.gpu.Btmp.shape[1] >= blks
        # 2-stage reduction - even if 1 block, as we have a += in second kernel
        self.fill_b_cuda(A0, A1, A2, w,
                         np.float32(Brenorm) if self.ftype == np.float32 else np.float64(
                             Brenorm),
                         sz, self.gpu.Btmp,
                         block=(1024, 1, 1),
                         grid=(blks, 1, 1),
                         stream=self.queue)
        self.fill_b_reduce_cuda(self.gpu.Btmp, B, np.int32(blks),
                                block=(1024, 1, 1),
                                grid=(1, 1, 1),
                                stream=self.queue)

    def error_reduce(self, addr, err_sum):
        # reference shape  (= GPU global dims)
        sh = err_sum.shape

        # stopper
        maxz = err_sum.shape[0]

        # batch buffers
        ferr = self.gpu.LLerr

        # print('maxz={}, ferr={}'.format(maxz, ferr.shape))
        assert(maxz <= np.prod(ferr.shape[:-2]))

        # Reduces the LL error along the last 2 dimensions.fd
        self.error_reduce_cuda(ferr, err_sum,
                               np.int32(ferr.shape[-2]),
                               np.int32(ferr.shape[-1]),
                               block=(32, 32, 1),
                               grid=(int(maxz), 1, 1),
                               stream=self.queue)

    def floating_intensity(self, addr, w, I, fic):

        # reference shape  (= GPU global dims)
        sh = I.shape

        # stopper
        maxz = I.shape[0]

        # internal buffers
        num = self.gpu.LLerr
        den = self.gpu.LLden
        Imodel = self.gpu.Imodel
        fic_tmp = self.gpu.fic_tmp

        ## math ##
        xall = np.int32(maxz * sh[1] * sh[2])
        bx = 1024

        self.floating_intensity_cuda_step1(Imodel, I, w, num, den,
                       xall,
                       block=(bx, 1, 1),
                       grid=(int((xall + bx - 1) // bx), 1, 1),
                       stream=self.queue)

        self.error_reduce_cuda(num, fic,
                               np.int32(num.shape[-2]),
                               np.int32(num.shape[-1]),
                               block=(32, 32, 1),
                               grid=(int(maxz), 1, 1),
                               stream=self.queue)

        self.error_reduce_cuda(den, fic_tmp,
                               np.int32(den.shape[-2]),
                               np.int32(den.shape[-1]),
                               block=(32, 32, 1),
                               grid=(int(maxz), 1, 1),
                               stream=self.queue)

        self.floating_intensity_cuda_step2(fic_tmp, fic, Imodel,
                       np.int32(Imodel.shape[-2]),
                       np.int32(Imodel.shape[-1]),
                       block=(32, 32, 1),
                       grid=(1, 1, int(maxz)),
                       stream=self.queue)


    def main(self, b_aux, addr, w, I):
        nmodes = self.nmodes
        # stopper
        maxz = I.shape[0]

        # batch buffers
        err = self.gpu.LLerr
        Imodel = self.gpu.Imodel
        aux = b_aux

        # write-to shape  (= GPU global dims)
        ish = aux.shape

        x = np.int32(ish[1] * ish[2])
        y = np.int32(nmodes)
        z = np.int32(maxz)
        bx = 1024

        #print(Imodel.dtype, I.dtype, w.dtype, err.dtype, aux.dtype, z, y, x)
        self.main_cuda(Imodel, I, w, err, aux,
                       z, y, x,
                       block=(bx, 1, 1),
                       grid=(int((x + bx - 1) // bx), 1, int(z)),
                       stream=self.queue)


class PoUpdateKernel(ab.PoUpdateKernel):

    def __init__(self, queue_thread=None, 
        math_type='float', accumulator_type='float'):
        super(PoUpdateKernel, self).__init__()
        # and now initialise the cuda
        if math_type not in ['double', 'float']:
            raise ValueError('only float and double are supported for math_type')
        if accumulator_type not in ['double', 'float']:
            raise ValueError('only float and double are supported for accumulator_type')
        self.math_type = math_type
        self.accumulator_type = accumulator_type
        self.queue = queue_thread
        self.norm = None
        self.MAK = MaxAbs2Kernel(self.queue)
        self.ob_update_cuda = load_kernel("ob_update", {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'MATH_TYPE': self.math_type
        })
        self.ob_update2_cuda = None  # load_kernel("ob_update2")
        self.pr_update_cuda = load_kernel("pr_update", {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'MATH_TYPE': self.math_type
        })
        self.pr_update2_cuda = None
        self.ob_update_ML_cuda = load_kernel("ob_update_ML", {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'MATH_TYPE': self.math_type
        })
        self.ob_update2_ML_cuda = None
        self.pr_update_ML_cuda = load_kernel("pr_update_ML", {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'MATH_TYPE': self.math_type
        })
        self.pr_update2_ML_cuda = None
        self.ob_update_local_cuda = load_kernel("ob_update_local", {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'MATH_TYPE': self.math_type,
            'ACC_TYPE': self.accumulator_type
        })
        self.pr_update_local_cuda = load_kernel("pr_update_local", {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'MATH_TYPE': self.math_type,
            'ACC_TYPE': self.accumulator_type
        })
        self.ob_norm_local_cuda = load_kernel("ob_norm_local", {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'MATH_TYPE': self.math_type,
            'ACC_TYPE': self.accumulator_type
        })
        self.pr_norm_local_cuda = load_kernel("pr_norm_local", {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'MATH_TYPE': self.math_type,
            'ACC_TYPE': self.accumulator_type
        })


    def ob_update(self, addr, ob, obn, pr, ex, atomics=True):
        obsh = [np.int32(ax) for ax in ob.shape]
        prsh = [np.int32(ax) for ax in pr.shape]
        if obn.dtype != np.float32:
            raise ValueError("Denominator must be float32 in current implementation")

        if atomics:
            if addr.shape[3] != 3 or addr.shape[2] != 5:
                raise ValueError('Address not in required shape for atomics ob_update')
            num_pods = np.int32(addr.shape[0] * addr.shape[1])
            self.ob_update_cuda(ex, num_pods, prsh[1], prsh[2],
                                pr, prsh[0], prsh[1], prsh[2],
                                ob, obsh[0], obsh[1], obsh[2],
                                addr,
                                obn,
                                block=(32, 32, 1), grid=(int(num_pods), 1, 1), stream=self.queue)
        else:
            if addr.shape[0] != 5 or addr.shape[1] != 3:
                raise ValueError('Address not in required shape for tiled ob_update')
            num_pods = np.int32(addr.shape[2] * addr.shape[3])
            if not self.ob_update2_cuda:
                self.ob_update2_cuda = load_kernel("ob_update2", {
                    "NUM_MODES": obsh[0],
                    "BDIM_X": 16,
                    "BDIM_Y": 16,
                    'IN_TYPE': 'float',
                    'OUT_TYPE': 'float',
                    'MATH_TYPE': self.math_type,
                    'ACC_TYPE': self.accumulator_type
                })

            grid = [int((x+15)//16) for x in ob.shape[-2:]]
            grid = (grid[1], grid[0], int(1))
            self.ob_update2_cuda(prsh[-1], obsh[0], num_pods, obsh[-2], obsh[-1],
                                 prsh[0],
                                 np.int32(ex.shape[0]),
                                 np.int32(ex.shape[1]),
                                 np.int32(ex.shape[2]),
                                 ob, obn, pr, ex, addr,
                                 block=(16, 16, 1), grid=grid, stream=self.queue)

    def pr_update(self, addr, pr, prn, ob, ex, atomics=True):
        obsh = [np.int32(ax) for ax in ob.shape]
        prsh = [np.int32(ax) for ax in pr.shape]
        if prn.dtype != np.float32:
            raise ValueError("Denominator must be float32 in current implementation")
        if atomics:
            if addr.shape[3] != 3 or addr.shape[2] != 5:
                raise ValueError('Address not in required shape for atomics pr_update')

            num_pods = np.int32(addr.shape[0] * addr.shape[1])
            self.pr_update_cuda(ex, num_pods, prsh[1], prsh[2],
                                pr, prsh[0], prsh[1], prsh[2],
                                ob, obsh[0], obsh[1], obsh[2],
                                addr,
                                prn,
                                block=(32, 32, 1), grid=(int(num_pods), 1, 1), stream=self.queue)
        else:
            if addr.shape[0] != 5 or addr.shape[1] != 3:
                raise ValueError('Address not in required shape for tiled pr_update')

            num_pods = np.int32(addr.shape[2] * addr.shape[3])
            if not self.pr_update2_cuda:
                self.pr_update2_cuda = load_kernel("pr_update2", {
                    "NUM_MODES": prsh[0],
                    "BDIM_X": 16,
                    "BDIM_Y": 16,
                    'IN_TYPE': 'float',
                    'OUT_TYPE': 'float',
                    'MATH_TYPE': self.math_type,
                    'ACC_TYPE': self.accumulator_type
                })

            grid = [int((x+15)//16) for x in pr.shape[-2:]]
            grid = (grid[0], grid[1], int(1))
            self.pr_update2_cuda(prsh[-1], obsh[-2], obsh[-1],
                                 prsh[0], obsh[0], num_pods,
                                 pr, prn, ob, ex, addr,
                                 block=(16, 16, 1), grid=grid, stream=self.queue)

    def ob_update_ML(self, addr, ob, pr, ex, fac=2.0, atomics=True):
        obsh = [np.int32(ax) for ax in ob.shape]
        prsh = [np.int32(ax) for ax in pr.shape]
        exsh = [np.int32(ax) for ax in ex.shape]

        if atomics:
            if addr.shape[3] != 3 or addr.shape[2] != 5:
                raise ValueError('Address not in required shape for tiled ob_update')

            num_pods = np.int32(addr.shape[0] * addr.shape[1])
            self.ob_update_ML_cuda(ex, num_pods, exsh[1], exsh[2],
                                   pr, prsh[0], prsh[1], prsh[2],
                                   ob, obsh[0], obsh[1], obsh[2],
                                   addr,
                                   np.float32(fac) if ex.dtype == np.complex64 else np.float64(fac),
                                   block=(32, 32, 1), grid=(int(num_pods), 1, 1), stream=self.queue)
        else:
            if addr.shape[0] != 5 or addr.shape[1] != 3:
                raise ValueError('Address not in required shape for tiled ob_update')

            num_pods = np.int32(addr.shape[2] * addr.shape[3])
            if not self.ob_update2_ML_cuda:
                self.ob_update2_ML_cuda = load_kernel("ob_update2_ML", {
                    "NUM_MODES": obsh[0],
                    "BDIM_X": 16,
                    "BDIM_Y": 16,
                    'IN_TYPE': 'float',
                    'OUT_TYPE': 'float',
                    'MATH_TYPE': self.math_type,
                    'ACC_TYPE': self.accumulator_type
                })
            grid = [int((x+15)//16) for x in ob.shape[-2:]]
            grid = (grid[1], grid[0], int(1))
            self.ob_update2_ML_cuda(prsh[-1], obsh[0], num_pods, obsh[-2], obsh[-1],
                                    prsh[0],
                                    np.int32(ex.shape[0]),
                                    np.int32(ex.shape[1]),
                                    np.int32(ex.shape[2]),
                                    ob, pr, ex, addr, 
                                    np.float32(fac) if ex.dtype == np.complex64 else np.float64(fac),
                                    block=(16, 16, 1), grid=grid, stream=self.queue)

    def pr_update_ML(self, addr, pr, ob, ex, fac=2.0, atomics=False):
        obsh = [np.int32(ax) for ax in ob.shape]
        prsh = [np.int32(ax) for ax in pr.shape]
        if atomics:
            if addr.shape[3] != 3 or addr.shape[2] != 5:
                raise ValueError('Address not in required shape for tiled pr_update')
            num_pods = np.int32(addr.shape[0] * addr.shape[1])
            self.pr_update_ML_cuda(ex, num_pods, prsh[1], prsh[2],
                                pr, prsh[0], prsh[1], prsh[2],
                                ob, obsh[0], obsh[1], obsh[2],
                                addr,
                                np.float32(fac) if ex.dtype == np.complex64 else np.float64(fac),
                                block=(32, 32, 1), grid=(int(num_pods), 1, 1), stream=self.queue)
        else:
            if addr.shape[0] != 5 or addr.shape[1] != 3:
                raise ValueError('Address not in required shape for tiled pr_update')
            num_pods = np.int32(addr.shape[2] * addr.shape[3])
            if not self.pr_update2_ML_cuda:
                self.pr_update2_ML_cuda = load_kernel("pr_update2_ML", {
                    "NUM_MODES": prsh[0],
                    "BDIM_X": 16,
                    "BDIM_Y": 16,
                    'IN_TYPE': 'float',
                    'OUT_TYPE': 'float',
                    'MATH_TYPE': self.math_type,
                    'ACC_TYPE': self.accumulator_type
                })

            grid = [int((x+15)//16) for x in pr.shape[-2:]]
            grid = (grid[0], grid[1], int(1))
            self.pr_update2_ML_cuda(prsh[-1], obsh[-2], obsh[-1],
                                 prsh[0], obsh[0], num_pods,
                                 pr, ob, ex, addr, 
                                 np.float32(fac) if ex.dtype == np.complex64 else np.float64(fac),
                                 block=(16, 16, 1), grid=grid, stream=self.queue)


    def ob_update_local(self, addr, ob, pr, ex, aux, prn, a=0., b=1.):
        prn_max = gpuarray.max(prn, stream=self.queue)
        obsh = [np.int32(ax) for ax in ob.shape]
        prsh = [np.int32(ax) for ax in pr.shape]
        exsh = [np.int32(ax) for ax in ex.shape]
        # atomics version only
        if addr.shape[3] != 3 or addr.shape[2] != 5:
            raise ValueError('Address not in required shape for tiled ob_update')
        num_pods = np.int32(addr.shape[0] * addr.shape[1])
        bx = 64
        by = 1
        self.ob_update_local_cuda(ex, aux,
            exsh[0], exsh[1], exsh[2],
            pr,
            prsh[0], prsh[1], prsh[2],
            prn,
            ob,
            obsh[0], obsh[1], obsh[2],
            addr,
            prn_max,
            np.float32(a),
            np.float32(b),
            block=(bx, by, 1),
            grid=(1, int((exsh[1] + by - 1)//by), int(num_pods)),
            stream=self.queue)

    def pr_update_local(self, addr, pr, ob, ex, aux, obn, obn_max, a=0., b=1.):
        obsh = [np.int32(ax) for ax in ob.shape]
        prsh = [np.int32(ax) for ax in pr.shape]
        exsh = [np.int32(ax) for ax in ex.shape]
        # atomics version only
        if addr.shape[3] != 3 or addr.shape[2] != 5:
            raise ValueError('Address not in required shape for tiled pr_update')
        num_pods = np.int32(addr.shape[0] * addr.shape[1])
        bx = 64
        by = 1
        self.pr_update_local_cuda(ex, aux,
            exsh[0], exsh[1], exsh[2],
            pr,
            prsh[0], prsh[1], prsh[2],
            obn,
            ob,
            obsh[0], obsh[1], obsh[2],
            addr,
            obn_max,
            np.float32(a),
            np.float32(b),
            block=(bx, by, 1),
            grid=(1, int((exsh[1] + by - 1) // by), int(num_pods)),
            stream=self.queue)

    def ob_norm_local(self, addr, ob, obn):
        obsh =  [np.int32(ax) for ax in ob.shape]
        obnsh = [np.int32(ax) for ax in obn.shape]
        bx = 64
        by = 1
        self.ob_norm_local_cuda(obn, 
            obnsh[0], obnsh[1], obnsh[2],
            ob,
            obsh[0], obsh[1], obsh[2],
            addr,
            block=(bx, by, 1),
            grid=(1, int((obnsh[1] + by - 1)//by), int(obnsh[0])),
            stream=self.queue)

    def pr_norm_local(self, addr, pr, prn):        
        prsh  = [np.int32(ax) for ax in pr.shape]
        prnsh = [np.int32(ax) for ax in prn.shape]
        bx = 64
        by = 1
        self.pr_norm_local_cuda(prn, 
            prnsh[0], prnsh[1], prnsh[2],
            pr,
            prsh[0], prsh[1], prsh[2],
            addr,
            block=(bx, by, 1),
            grid=(1, int((prnsh[1] + by - 1)//by), int(prnsh[0])),
            stream=self.queue)


class PositionCorrectionKernel(ab.PositionCorrectionKernel):
    from ptypy.accelerate.cuda_pycuda import address_manglers

    # these are used by the self.setup method - replacing them with the GPU implementation
    MANGLERS = {
        'Annealing': address_manglers.RandomIntMangler,
        'GridSearch': address_manglers.GridSearchMangler
    }

    def __init__(self, *args, queue_thread=None, math_type='float', accumulate_type='float', **kwargs):
        super(PositionCorrectionKernel, self).__init__(*args, **kwargs)
        # make sure we set the right stream in the mangler
        self.mangler.queue = queue_thread
        if math_type not in ['float', 'double']:
            raise ValueError('Only float or double math is supported')
        if accumulate_type not in ['float', 'double']:
            raise ValueError('Only float or double math is supported')
        
        # add kernels
        self.math_type = math_type
        self.accumulate_type = accumulate_type
        self.queue = queue_thread
        self._ob_shape = None
        self._ob_id = None
        self.fourier_error_cuda = load_kernel("fourier_error",{
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'MATH_TYPE': self.math_type
        })
        self.error_reduce_cuda = load_kernel("error_reduce", {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'BDIM_X': 32,
            'BDIM_Y': 32,
            'ACC_TYPE': self.accumulate_type
        })
        self.log_likelihood_cuda, self.log_likelihood_ml_cuda = load_kernel(
            ("log_likelihood", "log_likelihood_ml"), {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'MATH_TYPE': self.math_type
        }, "log_likelihood.cu")
        self.build_aux_pc_cuda = load_kernel("build_aux_position_correction", {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'MATH_TYPE': self.math_type
        })
        self.update_addr_and_error_state_cuda = load_kernel("update_addr_error_state", {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float'
        })

        self.gpu = Adict()
        self.gpu.fdev = None
        self.gpu.ferr = None

    def allocate(self):
        self.gpu.fdev = gpuarray.zeros(self.fshape, dtype=np.float32)
        self.gpu.ferr = gpuarray.zeros(self.fshape, dtype=np.float32)

    def build_aux(self, b_aux, addr, ob, pr):
        obr, obc = self._cache_object_shape(ob)
        sh = addr.shape
        nmodes = sh[1]
        maxz = sh[0]
        self.build_aux_pc_cuda(b_aux,
                               pr,
                               np.int32(pr.shape[1]), np.int32(pr.shape[2]),
                               ob,
                               obr, obc,
                               addr,
                               block=(32, 32, 1), grid=(int(maxz * nmodes), 1, 1), stream=self.queue)

    def fourier_error(self, f, addr, fmag, fmask, mask_sum):
        fdev = self.gpu.fdev
        ferr = self.gpu.ferr
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
        #import sys
        # float_size = sys.getsizeof(np.float32(4))
        # shared_memory_size =int(2 * 32 * 32 *float_size) # this doesn't work even though its the same...
        # shared_memory_size = int(49152)
        self.error_reduce_cuda(self.gpu.ferr,
                               err_fmag,
                               np.int32(self.fshape[1]),
                               np.int32(self.fshape[2]),
                               block=(32, 32, 1),
                               grid=(int(err_fmag.shape[0]), 1, 1),
                               stream=self.queue)

    def log_likelihood(self, b_aux, addr, mag, mask, err_phot):
        ferr = self.gpu.ferr
        self.log_likelihood_cuda(np.int32(self.nmodes),
                                 b_aux,
                                 mask,
                                 mag,
                                 addr,
                                 ferr,
                                 np.int32(self.fshape[1]),
                                 np.int32(self.fshape[2]),
                                 block=(32, 32, 1),
                                 grid=(int(mag.shape[0]), 1, 1),
                                 stream=self.queue)
        # TODO: we might want to move this call outside of here
        self.error_reduce(addr, err_phot)

    def log_likelihood_ml(self, b_aux, addr, I, weights, err_phot):
        ferr = self.gpu.ferr
        self.log_likelihood_ml_cuda(np.int32(self.nmodes),
                                 b_aux,
                                 weights,
                                 I,
                                 addr,
                                 ferr,
                                 np.int32(self.fshape[1]),
                                 np.int32(self.fshape[2]),
                                 block=(32, 32, 1),
                                 grid=(int(I.shape[0]), 1, 1),
                                 stream=self.queue)
        # TODO: we might want to move this call outside of here
        self.error_reduce(addr, err_phot)

    def update_addr_and_error_state(self, addr, error_state, mangled_addr, err_sum):
        # assume all data is on GPU!
        self.update_addr_and_error_state_cuda(addr, mangled_addr, error_state, err_sum,
            np.int32(addr.shape[1]),
            block=(32, 2, 1),
            grid=(1, int((err_sum.shape[0] + 1) // 2), 1),
            stream=self.queue)

    def _cache_object_shape(self, ob):
        oid = id(ob)

        if not oid == self._ob_id:
            self._ob_id = oid
            self._ob_shape = (np.int32(ob.shape[-2]), np.int32(ob.shape[-1]))

        return self._ob_shape

