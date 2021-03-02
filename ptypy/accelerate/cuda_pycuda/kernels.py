import numpy as np
from inspect import getfullargspec
from pycuda import gpuarray
from ptypy.utils.verbose import log, logger
from . import load_kernel
from ..base import kernels as ab
from ..base.kernels import Adict

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

        if self._fft_type=='cuda':
            try:
                from ptypy.accelerate.cuda_pycuda.cufft import FFT_cuda as FFT
            except:
                logger.warning('Unable to import cufft version - using Reikna instead')
                from ptypy.accelerate.cuda_pycuda.fft import FFT
        elif self._fft_type=='skcuda':
            try:
                from ptypy.accelerate.cuda_pycuda.cufft import FFT_skcuda as FFT
            except:
                logger.warning('Unable to import skcuda.fft version - using Reikna instead')
                from ptypy.accelerate.cuda_pycuda.fft import FFT
        else:
            from ptypy.accelerate.cuda_pycuda.fft import FFT

        if self.prop_type == 'farfield':
            self._fft1 = FFT(aux, self.queue,
                             pre_fft=self._p.pre_fft,
                             post_fft=self._p.post_fft,
                             symmetric=True,
                             forward=True)
            self._fft2 = FFT(aux, self.queue,
                             pre_fft=self._p.pre_ifft,
                             post_fft=self._p.post_ifft,
                             symmetric=True,
                             forward=False)
            self.fw = self._fft1.ft
            self.bw = self._fft2.ift
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
        self.fmag_all_update_cuda = load_kernel("fmag_all_update")
        self.fourier_error_cuda = load_kernel("fourier_error")
        self.fourier_error2_cuda = None
        self.error_reduce_cuda = load_kernel("error_reduce", {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'ACC_TYPE': self.accumulate_type
        })
        self.fourier_update_cuda = None
        self.log_likelihood_cuda = load_kernel("log_likelihood")
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

    def error_reduce(self, addr, err_sum):
        self.error_reduce_cuda(self.gpu.ferr,
                               err_sum,
                               np.int32(self.fshape[1]),
                               np.int32(self.fshape[2]),
                               block=(32, 32, 1),
                               grid=(int(err_sum.shape[0]), 1, 1),
                               shared=32*32*4,
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
        self.build_aux_cuda = load_kernel("build_aux", {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'MATH_TYPE': self.math_type
        })
        self.build_exit_cuda = load_kernel("build_exit", {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'MATH_TYPE': self.math_type
        })
        self.build_aux_no_ex_cuda = load_kernel("build_aux_no_ex", {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'MATH_TYPE': self.math_type
        })

    # DEPRECATED?
    def load(self, aux, ob, pr, ex, addr):
        super(AuxiliaryWaveKernel, self).load(aux, ob, pr, ex, addr)
        for key, array in self.npy.__dict__.items():
            self.ocl.__dict__[key] = gpuarray.to_gpu(array)

    def build_aux(self, b_aux, addr, ob, pr, ex, alpha=1.0):
        obr, obc = self._cache_object_shape(ob)
        # print('grid={}, 1, 1'.format(int(ex.shape[0])))
        # print('b_aux={}, sh={}'.format(type(b_aux), b_aux.shape))
        # print('ex={}, sh={}'.format(type(ex), ex.shape))
        # print('pr={}, sh={}'.format(type(pr), pr.shape))
        # print('ob={}, sh={}'.format(type(ob), ob.shape))
        # print('obr={}, obc={}'.format(obr, obc))
        # print('addr={}, sh={}'.format(type(addr), addr.shape))
        # print('stream={}'.format(self.queue))
        self.build_aux_cuda(b_aux,
                            ex,
                            np.int32(ex.shape[1]), np.int32(ex.shape[2]),
                            pr,
                            np.int32(ex.shape[1]), np.int32(ex.shape[2]),
                            ob,
                            obr, obc,
                            addr,
                            np.float32(alpha) if ex.dtype == np.complex64 else np.float64(alpha),
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
            # temporarily until all kernels are ported to be flexible with unified naming
            'CTYPE': 'complex<float>' if self.ctype == np.complex64 else 'complex<double>',
            'FTYPE': 'float' if self.ftype == np.float32 else 'double',
            'IN_TYPE': 'float' if self.ftype == np.float32 else 'double',
            'ACC_TYPE': self.accumulate_type,
            'MATH_TYPE': self.math_type
        }
        self.make_model_cuda = load_kernel('make_model', subs)
        self.make_a012_cuda = load_kernel('make_a012', subs)
        self.error_reduce_cuda = load_kernel('error_reduce', {
            **subs,
            'OUT_TYPE': 'float' if self.ftype == np.float32 else 'double'
        })
        self.fill_b_cuda = load_kernel('fill_b', {
            **subs, 
            'BDIM_X': 1024, 
            'OUT_TYPE': self.accumulate_type
        })
        self.fill_b_reduce_cuda = load_kernel(
            'fill_b_reduce', {
                **subs, 
                'BDIM_X': 1024, 
                'IN_TYPE': self.accumulate_type,  # must match out-type of fill_b
                'OUT_TYPE': 'float' if self.ftype == np.float32 else 'double'
            })
        self.main_cuda = load_kernel('gd_main', subs)
        self.floating_intensity_cuda_step1 = load_kernel('step1', subs,'intens_renorm.cu')
        self.floating_intensity_cuda_step2 = load_kernel('step2', subs,'intens_renorm.cu')

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
                               shared=32*32*4,
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
        x = np.int32(sh[1] * sh[2])
        z = np.int32(maxz)
        bx = 1024

        self.floating_intensity_cuda_step1(Imodel, I, w, num, den,
                       z, x,
                       block=(bx, 1, 1),
                       grid=(int((x + bx - 1) // bx), 1, int(z)),
                       stream=self.queue)

        self.error_reduce_cuda(num, fic,
                               np.int32(num.shape[-2]),
                               np.int32(num.shape[-1]),
                               block=(32, 32, 1),
                               grid=(int(maxz), 1, 1),
                               shared=32*32*4,
                               stream=self.queue)

        self.error_reduce_cuda(den, fic_tmp,
                               np.int32(den.shape[-2]),
                               np.int32(den.shape[-1]),
                               block=(32, 32, 1),
                               grid=(int(maxz), 1, 1),
                               shared=32*32*4,
                               stream=self.queue)

        self.floating_intensity_cuda_step2(fic_tmp, fic, Imodel,
                       z, x,
                       block=(bx, 1, 1),
                       grid=(int((x + bx - 1) // bx), 1, int(z)),
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
        self.ob_update2_cuda = None  # load_kernel("ob_update2")
        self.pr_update_cuda = load_kernel("pr_update", {
            'DENOM_TYPE': dtype
        })
        self.pr_update2_cuda = None
        self.ob_update_ML_cuda = load_kernel("ob_update_ML", {
            'CTYPE': 'complex<float>',
            'FTYPE': 'float'
        })
        self.ob_update2_ML_cuda = None
        self.pr_update_ML_cuda = load_kernel("pr_update_ML", {
            'CTYPE': 'complex<float>',
            'FTYPE': 'float'
        })
        self.pr_update2_ML_cuda = None

    def ob_update(self, addr, ob, obn, pr, ex, atomics=True):
        obsh = [np.int32(ax) for ax in ob.shape]
        prsh = [np.int32(ax) for ax in pr.shape]

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
                    'DENOM_TYPE': self.dtype
                })

            grid = [int((x+15)//16) for x in ob.shape[-2:]]
            grid = (grid[0], grid[1], int(1))
            self.ob_update2_cuda(prsh[-1], obsh[0], num_pods, obsh[-2],
                                 prsh[0],
                                 np.int32(ex.shape[0]),
                                 np.int32(ex.shape[1]),
                                 np.int32(ex.shape[2]),
                                 ob, obn, pr, ex, addr,
                                 block=(16, 16, 1), grid=grid, stream=self.queue)

    def pr_update(self, addr, pr, prn, ob, ex, atomics=True):
        obsh = [np.int32(ax) for ax in ob.shape]
        prsh = [np.int32(ax) for ax in pr.shape]
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
                    'DENOM_TYPE': self.dtype
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

        if atomics:
            if addr.shape[3] != 3 or addr.shape[2] != 5:
                raise ValueError('Address not in required shape for tiled ob_update')

            num_pods = np.int32(addr.shape[0] * addr.shape[1])
            self.ob_update_ML_cuda(ex, num_pods, prsh[1], prsh[2],
                                   pr, prsh[0], prsh[1], prsh[2],
                                   ob, obsh[0], obsh[1], obsh[2],
                                   addr,
                                   np.float32(fac),
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
                    'CTYPE': 'complex<float>',
                    'FTYPE': 'float'
                })
            grid = [int((x+15)//16) for x in ob.shape[-2:]]
            grid = (grid[0], grid[1], int(1))
            self.ob_update2_ML_cuda(prsh[-1], obsh[0], num_pods, obsh[-2],
                                    prsh[0],
                                    np.int32(ex.shape[0]),
                                    np.int32(ex.shape[1]),
                                    np.int32(ex.shape[2]),
                                    ob, pr, ex, addr, np.float32(fac),
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
                                np.float32(fac),
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
                    'CTYPE': 'complex<float>',
                    'FTYPE': 'float'
                })

            grid = [int((x+15)//16) for x in pr.shape[-2:]]
            grid = (grid[0], grid[1], int(1))
            self.pr_update2_ML_cuda(prsh[-1], obsh[-2], obsh[-1],
                                 prsh[0], obsh[0], num_pods,
                                 pr, ob, ex, addr, np.float32(fac),
                                 block=(16, 16, 1), grid=grid, stream=self.queue)


class PositionCorrectionKernel(ab.PositionCorrectionKernel):
    def __init__(self, aux, nmodes, queue_thread=None, math_type='float', accumulate_type='float'):
        super(PositionCorrectionKernel, self).__init__(aux, nmodes)
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
        self.fourier_error_cuda = load_kernel("fourier_error")
        self.error_reduce_cuda = load_kernel("error_reduce", {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'ACC_TYPE': self.accumulate_type
        })
        self.build_aux_pc_cuda = load_kernel("build_aux_position_correction", {
            'IN_TYPE': 'float',
            'OUT_TYPE': 'float',
            'MATH_TYPE': self.math_type
        })
        self.update_addr_and_error_state_cuda = load_kernel("update_addr_error_state")

        self.gpu = Adict()
        self.gpu.fdev = None
        self.gpu.ferr = None

    def allocate(self):
        self.gpu.fdev = gpuarray.zeros(self.fshape, dtype=np.float32)
        self.gpu.ferr = gpuarray.zeros(self.fshape, dtype=np.float32)

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
                               shared=32*32*4,
                               stream=self.queue)

    def update_addr_and_error_state_old(self, addr, error_state, mangled_addr, err_sum):
        '''
        updates the addresses and err state vector corresponding to the smallest error. I think this can be done on the cpu
        '''
        update_indices = err_sum < error_state
        log(4, "updating %s indices" % np.sum(update_indices))
        print('update ind {}, addr {}, mangled {}'.format(update_indices.shape, addr.shape, mangled_addr.shape))
        addr_cpu = addr.get_async(self.queue)
        self.queue.synchronize()
        addr_cpu[update_indices] = mangled_addr[update_indices]
        addr.set_async(ary=addr_cpu, stream=self.queue)

        error_state[update_indices] = err_sum[update_indices]

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

