import cupy as cp
import numpy as np

from ptypy.accelerate.cuda_common.utils import map2ctype
from ptypy.accelerate.base.array_utils import gaussian_kernel_2d
from ptypy.utils.math_utils import gaussian
from . import load_kernel


class ArrayUtilsKernel:
    def __init__(self, acc_dtype=cp.float64, queue=None):
        self.queue = queue
        self.acc_dtype = acc_dtype
        # Note: cupy's ReductionKernel is far less efficient
        self.cdot_cuda = load_kernel("dot", {
            'IN_TYPE': 'complex<float>',
            'ACC_TYPE': 'double' if acc_dtype == np.float64 else 'float'
        })
        self.dot_cuda = load_kernel("dot", {
            'IN_TYPE': 'float',
            'ACC_TYPE': 'double' if acc_dtype == np.float64 else 'float'
        })
        self.full_reduce_cuda = load_kernel("full_reduce", {
            'IN_TYPE': 'double' if acc_dtype == np.float64 else 'float',
            'OUT_TYPE': 'double' if acc_dtype == np.float64 else 'float',
            'ACC_TYPE': 'double' if acc_dtype == np.float64 else 'float',
            'BDIM_X': 1024
        })
        self.Ctmp = None

    def dot(self, A: cp.ndarray, B: cp.ndarray, out: cp.ndarray = None) -> cp.ndarray:
        assert A.dtype == B.dtype, "Input arrays must be of same data type"
        assert A.size == B.size, "Input arrays must be of the same size"

        if self.queue is not None:
            self.queue.use()
        if out is None:
            out = cp.empty(1, dtype=self.acc_dtype)

        block = (1024, 1, 1)
        grid = (int((B.size + 1023) // 1024), 1, 1)
        if self.acc_dtype == np.float32:
            elsize = 4
        elif self.acc_dtype == np.float64:
            elsize = 8
        if self.Ctmp is None or self.Ctmp.size < grid[0]:
            self.Ctmp = cp.zeros((grid[0],), dtype=self.acc_dtype)
        Ctmp = self.Ctmp
        if grid[0] == 1:
            Ctmp = out
        if np.iscomplexobj(B):
            self.cdot_cuda(grid, block, (A, B, np.int32(A.size), Ctmp),
                           shared_mem=1024 * elsize)
        else:
            self.dot_cuda(grid, block, (A, B, np.int32(A.size), Ctmp),
                          shared_mem=1024 * elsize)
        if grid[0] > 1:
            self.full_reduce_cuda((1, 1, 1), (1024, 1, 1), (self.Ctmp, out, np.int32(grid[0])),
                                  shared_mem=elsize*1024)

        return out

    def norm2(self, A, out=None):
        return self.dot(A, A, out)

class BatchedMultiplyKernel:
    def __init__(self, array, queue=None, math_type=np.complex64):
        self.queue = queue
        self.array_shape = array.shape[-2:]
        self.batches = int(np.prod(array.shape[0:array.ndim-2]) if array.ndim > 2 else 1)
        self.batched_multiply_cuda = load_kernel("batched_multiply", {
            'MPY_DO_SCALE': 'true',
            'MPY_DO_FILT': 'true',
            'IN_TYPE': 'float' if array.dtype==np.complex64 else 'double',
            'OUT_TYPE': 'float' if array.dtype==np.complex64 else 'double',
            'MATH_TYPE': 'float' if math_type==np.complex64 else 'double'
        })
        self.block = (32,32,1)
        self.grid = (
            int((self.array_shape[0] + 31) // 32),
            int((self.array_shape[1] + 31) // 32),
            int(self.batches)
        )

    def multiply(self, x,y, scale=1.):
        assert x.dtype == y.dtype, "Input arrays must be of same data type"
        assert x.shape[-2:] == y.shape[-2:], "Input arrays must be of the same size in last 2 dims"
        if self.queue is not None:
            self.queue.use()
        self.batched_multiply_cuda(self.grid,
                                   self.block,
                                   args=(x,x,y,
                                   np.float32(scale),
                                   np.int32(self.batches),
                                   np.int32(self.array_shape[0]),
                                   np.int32(self.array_shape[1])))

class TransposeKernel:

    def __init__(self, queue=None):
        self.queue = queue
        self.transpose_cuda = load_kernel("transpose", {
            'DTYPE': 'int',
            'BDIM': 16
        })

    def transpose(self, input, output):
        # only for int at the moment (addr array), and 2D (reshape pls)
        if len(input.shape) != 2:
            raise ValueError(
                "Only 2D tranpose is supported - reshape as desired")
        if input.shape[0] != output.shape[1] or input.shape[1] != output.shape[0]:
            raise ValueError("Input/Output must be of flipped shape")
        if input.dtype != np.int32 or output.dtype != np.int32:
            raise ValueError("Only int types are supported at the moment")

        width = input.shape[1]
        height = input.shape[0]
        blk = (16, 16, 1)
        grd = (
            int((input.shape[1] + 15) // 16),
            int((input.shape[0] + 15) // 16),
            1
        )
        if self.queue is not None:
            self.queue.use()
        self.transpose_cuda(
            grd, blk, (input, output, np.int32(width), np.int32(height)))


class MaxAbs2Kernel:

    def __init__(self, queue=None):
        self.queue = queue
        # we lazy-load this depending on the data types we get
        self.max_abs2_cuda = {}

    def max_abs2(self, X: cp.ndarray, out: cp.ndarray):
        """ Calculate max(abs(x)**2) across the final 2 dimensions"""
        rows = np.int32(X.shape[-2])
        cols = np.int32(X.shape[-1])
        firstdims = np.int32(np.prod(X.shape[:-2]))
        gy = int(rows)
        # lazy-loading, keeping scratch memory and both kernels in the same dictionary
        bx = int(64)
        version = '{},{},{}'.format(
            map2ctype(X.dtype), map2ctype(out.dtype), gy)
        if version not in self.max_abs2_cuda:
            step1, step2 = load_kernel(
                ("max_abs2_step1", "max_abs2_step2"),
                {
                    'IN_TYPE': map2ctype(X.dtype),
                    'OUT_TYPE': map2ctype(out.dtype),
                    'BDIM_X': bx,
                }, "max_abs2.cu")
            self.max_abs2_cuda[version] = {
                'step1': step1,
                'step2': step2,
                'scratchmem': cp.empty((gy,), dtype=out.dtype)
            }

        # if self.max_abs2_cuda[version]['scratchmem'] is None \
        #     or self.max_abs2_cuda[version]['scratchmem'].shape[0] != gy:
        #     self.max_abs2_cuda[version]['scratchmem'] =
        scratch = self.max_abs2_cuda[version]['scratchmem']

        if self.queue is not None:
            self.queue.use()

        self.max_abs2_cuda[version]['step1'](
            (1, gy, 1), (bx, 1, 1), (X, firstdims, rows, cols, scratch))
        self.max_abs2_cuda[version]['step2'](
            (1, 1, 1), (bx, 1, 1), (scratch, np.int32(gy), out))


class CropPadKernel:

    def __init__(self, queue=None):
        self.queue = queue
        # we lazy-load this depending on the data types we get
        self.fill3D_cuda = {}

    def fill3D(self, A, B, offset=[0, 0, 0]):
        """
        Fill 3-dimensional array A with B.
        """
        if A.ndim < 3 or B.ndim < 3:
            raise ValueError('Input arrays must each be at least 3D')
        assert A.ndim == B.ndim, "Input and Output must have the same number of dimensions."
        ash = A.shape
        bsh = B.shape
        misfit = np.array(bsh) - np.array(ash)
        assert not misfit[:-3].any(
        ), "Input and Output must have the same shape everywhere but the last three axes."

        Alim = np.array(A.shape[-3:])
        Blim = np.array(B.shape[-3:])
        off = np.array(offset)
        Ao = off.copy()
        Ao[Ao < 0] = 0
        Bo = -off.copy()
        Bo[Bo < 0] = 0
        assert (Bo < Blim).all() and (Ao < Alim).all(
        ), "At least one dimension lacks overlap"
        Ao = Ao.astype(np.int32)
        Bo = Bo.astype(np.int32)
        lengths = np.array([
            min(off[0] + Blim[0], Alim[0]) - Ao[0],
            min(off[1] + Blim[1], Alim[1]) - Ao[1],
            min(off[2] + Blim[2], Alim[2]) - Ao[2],
        ], dtype=np.int32)
        lengths2 = np.array([
            min(Alim[0] - off[0], Blim[0]) - Bo[0],
            min(Alim[1] - off[1], Blim[1]) - Bo[1],
            min(Alim[2] - off[2], Blim[2]) - Bo[2],
        ], dtype=np.int32)
        assert (lengths == lengths2).all(
        ), "left and right lenghts are not matching"
        batch = int(np.prod(A.shape[:-3]))

        # lazy loading depending on data type
        version = '{},{}'.format(map2ctype(B.dtype), map2ctype(A.dtype))
        if version not in self.fill3D_cuda:
            self.fill3D_cuda[version] = load_kernel("fill3D", {
                'IN_TYPE': map2ctype(B.dtype),
                'OUT_TYPE': map2ctype(A.dtype)
            })
        bx = by = 32
        if self.queue is not None:
            self.queue.use()
        self.fill3D_cuda[version](
            (int((lengths[2] + bx - 1)//bx),
             int((lengths[1] + by - 1)//by),
             int(batch)),
            (int(bx), int(by), int(1)),
            (A, B,
             np.int32(A.shape[-3]), np.int32(A.shape[-2]
                                             ), np.int32(A.shape[-1]),
             np.int32(B.shape[-3]), np.int32(B.shape[-2]
                                             ), np.int32(B.shape[-1]),
             Ao[0], Ao[1], Ao[2],
             Bo[0], Bo[1], Bo[2],
             lengths[0], lengths[1], lengths[2])
        )

    def crop_pad_2d_simple(self, A, B):
        """
        Places B in A centered around the last two axis. A and B must be of the same shape
        anywhere but the last two dims.
        """
        assert A.ndim >= 2, "Arrays must have more than 2 dimensions."
        assert A.ndim == B.ndim, "Input and Output must have the same number of dimensions."
        misfit = np.array(A.shape) - np.array(B.shape)
        assert not misfit[:-2].any(
        ), "Input and Output must have the same shape everywhere but the last two axes."
        if A.ndim == 2:
            A = A.reshape((1,) + A.shape)
        if B.ndim == 2:
            B = B.reshape((1,) + B.shape)
        a1, a2 = A.shape[-2:]
        b1, b2 = B.shape[-2:]
        offset = [0, a1 // 2 - b1 // 2, a2 // 2 - b2 // 2]
        self.fill3D(A, B, offset)


class DerivativesKernel:
    def __init__(self, dtype, queue=None):
        if dtype == np.float32:
            stype = "float"
        elif dtype == np.complex64:
            stype = "complex<float>"
        else:
            raise NotImplementedError(
                "delxf is only implemented for float32 and complex64")

        self.queue = queue
        self.dtype = dtype
        self.last_axis_block = (256, 4, 1)
        self.mid_axis_block = (256, 4, 1)

        self.delxf_last, self.delxf_mid = load_kernel(
            ("delx_last", "delx_mid"),
            file="delx.cu",
            subs={
                'IS_FORWARD': 'true',
                'BDIM_X': str(self.last_axis_block[0]),
                'BDIM_Y': str(self.last_axis_block[1]),
                'IN_TYPE': stype,
                'OUT_TYPE': stype
            })
        self.delxb_last, self.delxb_mid = load_kernel(
            ("delx_last", "delx_mid"),
            file="delx.cu",
            subs={
                'IS_FORWARD': 'false',
                'BDIM_X': str(self.last_axis_block[0]),
                'BDIM_Y': str(self.last_axis_block[1]),
                'IN_TYPE': stype,
                'OUT_TYPE': stype
            })

    def delxf(self, input, out, axis=-1):
        if input.dtype != self.dtype:
            raise ValueError('Invalid input data type')

        if axis < 0:
            axis = input.ndim + axis
        axis = np.int32(axis)

        if self.queue is not None:
            self.queue.use()

        if axis == input.ndim - 1:
            flat_dim = np.int32(np.prod(input.shape[0:-1]))
            self.delxf_last((
                int((flat_dim +
                     self.last_axis_block[1] - 1) // self.last_axis_block[1]),
                1, 1),
                self.last_axis_block, (input, out, flat_dim, np.int32(input.shape[axis])))
        else:
            lower_dim = np.int32(np.prod(input.shape[(axis+1):]))
            higher_dim = np.int32(np.prod(input.shape[:axis]))
            gx = int(
                (lower_dim + self.mid_axis_block[0] - 1) // self.mid_axis_block[0])
            gy = 1
            gz = int(higher_dim)
            self.delxf_mid((gx, gy, gz), self.mid_axis_block, (input,
                           out, lower_dim, higher_dim, np.int32(input.shape[axis])))

    def delxb(self, input, out, axis=-1):
        if input.dtype != self.dtype:
            raise ValueError('Invalid input data type')

        if axis < 0:
            axis = input.ndim + axis
        axis = np.int32(axis)

        if self.queue is not None:
            self.queue.use()
        if axis == input.ndim - 1:
            flat_dim = np.int32(np.prod(input.shape[0:-1]))
            self.delxb_last((
                int((flat_dim +
                     self.last_axis_block[1] - 1) // self.last_axis_block[1]),
                1, 1), self.last_axis_block, (input, out, flat_dim, np.int32(input.shape[axis])))
        else:
            lower_dim = np.int32(np.prod(input.shape[(axis+1):]))
            higher_dim = np.int32(np.prod(input.shape[:axis]))
            gx = int(
                (lower_dim + self.mid_axis_block[0] - 1) // self.mid_axis_block[0])
            gy = 1
            gz = int(higher_dim)
            self.delxb_mid((gx, gy, gz), self.mid_axis_block, (input,
                           out, lower_dim, higher_dim, np.int32(input.shape[axis])))


class FFTGaussianSmoothingKernel:
    def __init__(self, queue=None, kernel_type='float'):
        if kernel_type not in ['float', 'double']:
            raise ValueError('Invalid data type for kernel')
        self.kernel_type = kernel_type
        self.dtype = np.complex64
        self.stype = "complex<float>"
        self.queue = queue
        self.sigma = None

        from .kernels import FFTFilterKernel

        # Create general FFT filter object 
        self.fft_filter = FFTFilterKernel(queue_thread=queue)
        
    def allocate(self, shape, sigma=1.):

        # Create kernel
        self.sigma = sigma
        kernel = self._compute_kernel(shape, sigma)

        # Allocate filter
        kernel_dev = cp.asarray(kernel)
        self.fft_filter.allocate(kernel=kernel_dev)

    def _compute_kernel(self, shape, sigma):
        # Create kernel
        sigma = np.array(sigma)
        if sigma.size == 1:
            sigma = np.array([sigma, sigma])
        if sigma.size > 2:
            raise NotImplementedError("Only batches of 2D arrays allowed!")
        self.sigma = sigma

        kernel = gaussian_kernel_2d(shape, sigma[0], sigma[1]).astype(self.dtype)
        
        # Extend kernel if needed
        if len(shape) == 3:
            kernel = np.tile(kernel, (shape[0],1,1))

        return kernel

    def filter(self, data, sigma=None):
        """
        Apply filter in-place
        
        If sigma is not None: reallocate a new fft filter first.
        """
        if self.sigma is None:
            self.allocate(shape=data.shape, sigma=sigma)
        else:
            self.fft_filter.set_kernel(self._compute_kernel(data.shape, sigma))
        
        self.fft_filter.apply_filter(data)


class GaussianSmoothingKernel:
    def __init__(self, queue=None, num_stdevs=4, kernel_type='float'):
        if kernel_type not in ['float', 'double']:
            raise ValueError('Invalid data type for kernel')
        self.kernel_type = kernel_type
        self.dtype = np.complex64
        self.stype = "complex<float>"
        self.queue = queue
        self.num_stdevs = num_stdevs
        self.blockdim_x = 4
        self.blockdim_y = 16

        # At least 2 blocks per SM
        self.max_shared_per_block = 48 * 1024 // 2
        self.max_shared_per_block_complex = self.max_shared_per_block / \
            2 * np.dtype(np.float32).itemsize
        self.max_kernel_radius = int(
            self.max_shared_per_block_complex / self.blockdim_y)

        self.convolution_row = load_kernel(
            "convolution_row", file="convolution.cu", subs={
                'BDIM_X': self.blockdim_x,
                'BDIM_Y': self.blockdim_y,
                'DTYPE': self.stype,
                'MATH_TYPE': self.kernel_type
            })
        self.convolution_col = load_kernel(
            "convolution_col", file="convolution.cu", subs={
                'BDIM_X': self.blockdim_y,   # NOTE: we swap x and y in this columns
                'BDIM_Y': self.blockdim_x,
                'DTYPE': self.stype,
                'MATH_TYPE': self.kernel_type
            })
        # pre-allocate kernel memory on gpu, with max-radius to accomodate
        dtype = np.float32 if self.kernel_type == 'float' else np.float64
        self.kernel_gpu = cp.empty((self.max_kernel_radius,), dtype=dtype)
        # keep track of previus radius and std to determine if we need to transfer again
        self.r = 0
        self.std = 0

    def convolution(self, data, mfs, tmp=None):
        """
        Calculates a stacked 2D convolution for smoothing, with the standard deviations
        given in mfs (stdx, stdy). It works in-place in the data array,
        and tmp is a gpu-allocated array of the same size and type as data,
        used internally for temporary storage
        """
        ndims = data.ndim
        shape = data.shape

        # Create temporary array (if not given)
        if tmp is None:
            tmp = cp.empty(shape, dtype=data.dtype)
        assert shape == tmp.shape and data.dtype == tmp.dtype

        # Check input dimensions
        if ndims == 3:
            batches, y, x = shape
            stdy, stdx = mfs
        elif ndims == 2:
            batches = 1
            y, x = shape
            stdy, stdx = mfs
        elif ndims == 1:
            batches = 1
            y, x = shape[0], 1
            stdy, stdx = mfs[0], 0.0
        else:
            raise NotImplementedError(
                "input needs to be of dimensions 0 < ndims <= 3")

        input = data
        output = tmp

        if self.queue is not None:
            self.queue.use()

        # Row convolution kernel
        # TODO: is this threshold acceptable in all cases?
        if stdx > 0.1:
            r = int(self.num_stdevs * stdx + 0.5)
            if r > self.max_kernel_radius:
                raise ValueError("Size of Gaussian kernel too large")
            if r != self.r or stdx != self.std:
                # recalculate + transfer
                g = gaussian(np.arange(-r, r+1), stdx)
                g /= g.sum()
                k = np.ascontiguousarray(g[r:].astype(
                    np.float32 if self.kernel_type == 'float' else np.float64))
                self.kernel_gpu[:r+1] = cp.asarray(k[:])
                self.r = r
                self.std = stdx

            bx = self.blockdim_x
            by = self.blockdim_y

            shared = (bx + 2*r) * by * np.dtype(np.complex64).itemsize
            if shared > self.max_shared_per_block:
                raise MemoryError("Cannot run kernel in shared memory")

            blk = (bx, by, 1)
            grd = (int((y + bx - 1) // bx), int((x + by-1) // by), batches)
            self.convolution_row(grd, blk, (input, output, np.int32(y), np.int32(x), self.kernel_gpu, np.int32(r)),
                                 shared_mem=shared)

            input = output
            output = data

        # Column convolution kernel
        # TODO: is this threshold acceptable in all cases?
        if stdy > 0.1:
            r = int(self.num_stdevs * stdy + 0.5)
            if r > self.max_kernel_radius:
                raise ValueError("Size of Gaussian kernel too large")
            if r != self.r or stdy != self.std:
                # recalculate + transfer
                g = gaussian(np.arange(-r, r+1), stdy)
                g /= g.sum()
                k = np.ascontiguousarray(g[r:].astype(
                    np.float32 if self.kernel_type == 'float' else np.float64))
                self.kernel_gpu[:r+1] = cp.asarray(k[:])
                self.r = r
                self.std = stdy

            bx = self.blockdim_y
            by = self.blockdim_x

            shared = (by + 2*r) * bx * np.dtype(np.complex64).itemsize
            if shared > self.max_shared_per_block:
                raise MemoryError("Cannot run kernel in shared memory")

            blk = (bx, by, 1)
            grd = (int((y + bx - 1) // bx), int((x + by-1) // by), batches)
            self.convolution_col(grd, blk, (input, output, np.int32(y), np.int32(x), self.kernel_gpu, np.int32(r)),
                                 shared_mem=shared)

        # TODO: is this threshold acceptable in all cases?
        if (stdx <= 0.1 and stdy <= 0.1):
            return   # nothing to do
        elif (stdx > 0.1 and stdy > 0.1):
            return   # both parts have run, output is back in data
        else:
            data[:] = tmp[:]  # only one of them has run, output is in tmp


class ClipMagnitudesKernel:

    def __init__(self, queue=None):
        self.queue = queue
        self.clip_magnitudes_cuda = load_kernel("clip_magnitudes", {
            'IN_TYPE': 'complex<float>',
        })

    def clip_magnitudes_to_range(self, array, clip_min, clip_max):
        if self.queue is not None:
            self.queue.use()

        cmin = np.float32(clip_min)
        cmax = np.float32(clip_max)

        npixel = np.int32(np.prod(array.shape))
        bx = 256
        gx = int((npixel + bx - 1) // bx)
        self.clip_magnitudes_cuda((gx, 1, 1), (bx, 1, 1), (array, cmin, cmax,
                npixel))

class MassCenterKernel:

    def __init__(self, queue=None):
        self.queue = queue
        self.threadsPerBlock = 256

        self.indexed_sum_middim_cuda = load_kernel("indexed_sum_middim",
                file="mass_center.cu", subs={
                    'IN_TYPE': 'float',
                    'BDIM_X' : self.threadsPerBlock,
                    'BDIM_Y' : 1,
                    }
                )

        self.indexed_sum_lastdim_cuda = load_kernel("indexed_sum_lastdim",
                file="mass_center.cu", subs={
                    'IN_TYPE': 'float',
                    'BDIM_X' : 32,
                    'BDIM_Y' : 32,
                    }
                )

        self.final_sums_cuda = load_kernel("final_sums",
                file="mass_center.cu", subs={
                    'IN_TYPE': 'float',
                    'BDIM_X' : 256,
                    'BDIM_Y' : 1,
                    }
                )

    def mass_center(self, array):
        if array.dtype != np.float32:
            raise NotImplementedError("mass_center is only implemented for float32")

        i = np.int32(array.shape[0])
        m = np.int32(array.shape[1])
        if array.ndim >= 3:
            n = np.int32(array.shape[2])
        else:
            n = np.int32(1)

        if self.queue is not None:
            self.queue.use()

        total_sum = cp.sum(array, dtype=np.float32).get()
        sc = np.float32(1. / total_sum.item())

        i_sum = cp.empty(array.shape[0], dtype=np.float32)
        m_sum = cp.empty(array.shape[1], dtype=np.float32)
        n_sum = cp.empty(int(n), dtype=np.float32)
        out = cp.empty(3 if n>1 else 2, dtype=np.float32)

        # sum all dims except the first, multiplying by the index and scaling factor
        block_ = (self.threadsPerBlock, 1, 1)
        grid_ = (int(i), 1, 1)
        self.indexed_sum_middim_cuda(grid_, block_, (array, i_sum, np.int32(1), i, n*m, sc),
                shared_mem=self.threadsPerBlock*4)

        if array.ndim >= 3:
            # 3d case
            # sum all dims, except the middle, multiplying by the index and scaling factor
            block_ = (self.threadsPerBlock, 1, 1)
            grid_ = (int(m), 1, 1)
            self.indexed_sum_middim_cuda(grid_, block_, (array, m_sum, i, n, m, sc),
                    shared_mem=self.threadsPerBlock*4)

            # sum the all dims except the last, multiplying by the index and scaling factor
            block_ = (32, 32, 1)
            grid_ = (1, int(n + 32 - 1) // 32, 1)
            self.indexed_sum_lastdim_cuda(grid_, block_, (array, n_sum, i*m, n, sc),
                    shared_mem=32*32*4)
        else:
            # 2d case
            # sum the all dims except the last, multiplying by the index and scaling factor
            block_ = (32, 32, 1)
            grid_ = (1, int(m + 32 - 1) // 32, 1)
            self.indexed_sum_lastdim_cuda(grid_, block_, (array, m_sum, i, m, sc),
                    shared_mem=32*32*4)

        block_ = (256, 1, 1)
        grid_ = (3 if n>1 else 2, 1, 1)
        self.final_sums_cuda(grid_, block_, (i_sum, i, m_sum, m, n_sum, n, out),
                shared_mem=256*4)

        return out

class Abs2SumKernel:

    def __init__(self, dtype, queue=None):
        self.in_stype = map2ctype(dtype)
        if self.in_stype == 'complex<float>':
            self.out_stype = 'float'
            self.out_dtype = np.float32
        elif self.in_stype == 'copmlex<double>':
            self.out_stype = 'double'
            self.out_dtype = np.float64
        else:
            self.out_stype = self.in_stype
            self.out_dtype = dtype

        self.queue = queue
        self.threadsPerBlock = 32

        self.abs2sum_cuda = load_kernel("abs2sum", subs={
                    'IN_TYPE': self.in_stype,
                    'OUT_TYPE' : self.out_stype,
                    'BDIM_X' : 32,
                    }
                )

    def abs2sum(self, array):
        nmodes = np.int32(array.shape[0])
        row, col = array.shape[1:]
        out = cp.empty(array.shape[1:], dtype=self.out_dtype)

        if self.queue is not None:
            self.queue.use()
        block_ = (32, 1, 1)
        grid_ = (1, row, 1)
        self.abs2sum_cuda(grid_, block_, (array, nmodes, np.int32(row), np.int32(col), out))

        return out

class InterpolatedShiftKernel:

    def __init__(self, queue=None):
        self.queue = queue

        self.integer_shift_cuda, self.linear_interpolate_cuda = load_kernel(
                ("integer_shift_kernel", "linear_interpolate_kernel"),
                file="interpolated_shift.cu", subs={
                    'IN_TYPE': 'complex<float>',
                    'OUT_TYPE': 'complex<float>',
                    'BDIM_X' : 32,
                    'BDIM_Y' : 32,
                    }
                )

    def interpolate_shift(self, array, shift):
        shift = np.asarray(shift, dtype=np.float32)
        if len(shift) != 2:
            raise NotImplementedError("Shift only applied to 2D array.")
        if array.dtype != np.complex64:
            raise NotImplementedError("Only complex single precision supported")
        if array.ndim == 3:
            items, rows, columns = array.shape
        elif array.ndim == 2:
            items, rows, columns = 1, *array.shape
        else:
            raise NotImplementedError("Only 2- or 3-dimensional arrays supported")

        offsetRow, offsetCol = shift

        offsetRowFrac, offsetRowInt = np.modf(offsetRow)
        offsetColFrac, offsetColInt = np.modf(offsetCol)

        if self.queue is not None:
            self.queue.use()

        out = cp.empty_like(array)
        block_ = (32, 32, 1)
        grid_ = ((rows + 31) // 32, (columns + 31) // 32, items)

        if np.abs(offsetRowFrac) < 1e-6 and np.abs(offsetColFrac) < 1e-6:
            if offsetRowInt == 0 and offsetColInt == 0:
                # no transformation at all
                out = array
            else:
                # no fractional part, so we can just use a shifted copy
                self.integer_shift_cuda(grid_, block_, (array, out, np.int32(rows),
                        np.int32(columns), np.int32(offsetRow),
                        np.int32(offsetCol)))
        else:
            self.linear_interpolate_cuda(grid_, block_, (array, out, np.int32(rows),
                    np.int32(columns), np.float32(offsetRow),
                    np.float32(offsetCol)),
                    shared_mem=(32+2)**2*8+32*(32+2)*8)

        return out

