from . import load_kernel
from pycuda import gpuarray
from ptypy.utils import gaussian
import numpy as np

class ArrayUtilsKernel:
    def __init__(self, acc_dtype=np.float64, queue=None):
        self.queue = queue
        self.acc_dtype = acc_dtype
        self.cdot_cuda = load_kernel("dot", {
            'INTYPE': 'complex<float>',
            'ACCTYPE': 'double' if acc_dtype==np.float64 else 'float'
        })
        self.dot_cuda = load_kernel("dot", {
            'INTYPE': 'float',
            'ACCTYPE': 'double' if acc_dtype==np.float64 else 'float'
        })
        self.full_reduce_cuda = load_kernel("full_reduce", {
            'DTYPE': 'double' if acc_dtype==np.float64 else 'float',
            'BDIM_X': 1024
        })
        self.transpose_cuda = load_kernel("transpose", {
            'DTYPE': 'int',
            'BDIM': 16
        })
        self.Ctmp = None
        
    def dot(self, A, B, out=None):
        assert A.dtype == B.dtype, "Input arrays must be of same data type"
        assert A.size == B.size, "Input arrays must be of the same size"
        
        if out is None:
            out = gpuarray.zeros((1,), dtype=self.acc_dtype)

        block = (1024, 1, 1)
        grid = (int((B.size + 1023) // 1024), 1, 1)
        if self.acc_dtype == np.float32:
            elsize = 4
        elif self.acc_dtype == np.float64:
            elsize = 8
        if self.Ctmp is None or self.Ctmp.size < grid[0]:
            self.Ctmp = gpuarray.zeros((grid[0],), dtype=self.acc_dtype)
        Ctmp = self.Ctmp
        if grid[0] == 1:
            Ctmp = out
        if np.iscomplexobj(B):
            self.cdot_cuda(A, B, np.int32(A.size), Ctmp,
                block=block, grid=grid, 
                shared=1024 * elsize,
                stream=self.queue)
        else:
            self.dot_cuda(A, B, np.int32(A.size), Ctmp,
                block=block, grid=grid,
                shared=1024 * elsize,
                stream=self.queue)
        if grid[0] > 1:
            self.full_reduce_cuda(self.Ctmp, out, np.int32(grid[0]), 
                block=(1024, 1, 1), grid=(1,1,1), shared=elsize*1024,
                stream=self.queue)
        
        return out

    def transpose(self, input, output):
        # only for int at the moment (addr array), and 2D (reshape pls)
        if len(input.shape) != 2:
            raise ValueError("Only 2D tranpose is supported - reshape as desired")
        if input.shape[0] != output.shape[1] or input.shape[1] != output.shape[0]:
            raise ValueError("Input/Output must be of flipped shape")
        if input.dtype != np.int32 or output.dtype != np.int32:
            raise ValueError("Only int types are supported at the moment")
        
        width = input.shape[1]
        height = input.shape[0]
        blk = (16, 16, 1)
        grd = (
            int((input.shape[1] + 15)// 16),
            int((input.shape[0] + 15)// 16),
            1
        )
        self.transpose_cuda(input, output, np.int32(width), np.int32(height),
            block=blk, grid=grd, stream=self.queue)

    def norm2(self, A, out=None):
        return self.dot(A, A, out)


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
                int((flat_dim +
                     self.last_axis_block[1] - 1) // self.last_axis_block[1]),
                1, 1),
                stream=self.queue
            )
        else:
            lower_dim = np.int32(np.product(input.shape[(axis+1):]))
            higher_dim = np.int32(np.product(input.shape[:axis]))
            gx = int(
                (lower_dim + self.mid_axis_block[0] - 1) // self.mid_axis_block[0])
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
                int((flat_dim +
                     self.last_axis_block[1] - 1) // self.last_axis_block[1]),
                1, 1),
                stream=self.queue
            )
        else:
            lower_dim = np.int32(np.product(input.shape[(axis+1):]))
            higher_dim = np.int32(np.product(input.shape[:axis]))
            gx = int(
                (lower_dim + self.mid_axis_block[0] - 1) // self.mid_axis_block[0])
            gy = 1
            gz = int(higher_dim)
            self.delxb_mid(input, out, lower_dim, higher_dim, np.int32(input.shape[axis]),
                           block=self.mid_axis_block,
                           grid=(gx, gy, gz),
                           stream=self.queue
                           )


class GaussianSmoothingKernel:
    def __init__(self, queue=None, num_stdevs=4):
        self.dtype = np.complex64
        self.stype = "complex<float>"
        self.queue = queue
        self.num_stdevs = num_stdevs
        self.blockdim_x = 4
        self.blockdim_y = 16
        
        # At least 2 blocks per SM
        self.max_shared_per_block = 48 * 1024 // 2 
        self.max_shared_per_block_complex = self.max_shared_per_block / 2 * np.dtype(np.float32).itemsize
        self.max_kernel_radius = self.max_shared_per_block_complex / self.blockdim_y

        self.convolution_row = load_kernel("convolution_row", file="convolution.cu", subs={
            'BDIM_X': self.blockdim_x,
            'BDIM_Y': self.blockdim_y,
            'DTYPE': self.stype
        })
        self.convolution_col = load_kernel("convolution_col", file="convolution.cu", subs={
            'BDIM_X': self.blockdim_y,
            'BDIM_Y': self.blockdim_x,
            'DTYPE': self.stype
        })

    
    def convolution(self, input, output, mfs):
        ndims = input.ndim
        shape = input.shape

        # Check input dimensions        
        if ndims == 3:
            batches,y,x = shape
            stdy, stdx = mfs
        elif ndims == 2:
            batches = 1
            y,x = shape
            stdy, stdx = mfs
        elif ndims == 1:
            batches = 1
            y,x = shape[0],1
            stdy, stdx = mfs[0], 0.0
        else:
            raise NotImplementedError("input needs to be of dimensions 0 < ndims <= 3")

        # Row convolution kernel
        # TODO: is this threshold acceptable in all cases?
        if stdx > 0.1:
            r = int(self.num_stdevs * stdx + 0.5)
            g = gaussian(np.arange(-r,r+1), stdx)
            g /= g.sum()
            kernel = gpuarray.to_gpu(g[r:].astype(np.float32))
            if r > self.max_kernel_radius:
                raise ValueError("Size of Gaussian kernel too large")

            bx = self.blockdim_x
            by = self.blockdim_y
            
            shared = (bx + 2*r) * by * np.dtype(np.complex64).itemsize
            if shared > self.max_shared_per_block:
                raise MemoryError("Cannot run kernel in shared memory")

            blk = (bx, by, 1)
            grd = (int((y + bx -1)// bx), int((x + by-1)// by), batches)
            self.convolution_row(input, output, np.int32(y), np.int32(x), kernel, np.int32(r), 
                                 block=blk, grid=grd, shared=shared, stream=self.queue)

            # Overwrite input
            input = output

        # Column convolution kernel
        # TODO: is this threshold acceptable in all cases?
        if stdy > 0.1:
            r = int(self.num_stdevs * stdy + 0.5)
            g = gaussian(np.arange(-r,r+1), stdy)
            g /= g.sum()
            kernel = gpuarray.to_gpu(g[r:].astype(np.float32))
            if r > self.max_kernel_radius:
                raise ValueError("Size of Gaussian kernel too large")

            bx = self.blockdim_y
            by = self.blockdim_x
            
            shared = (by + 2*r) * bx * np.dtype(np.complex64).itemsize
            if shared > self.max_shared_per_block:
                raise MemoryError("Cannot run kernel in shared memory")

            blk = (bx, by, 1)
            grd = (int((y + bx -1)// bx), int((x + by-1)// by), batches)
            self.convolution_col(input, output, np.int32(y), np.int32(x), kernel, np.int32(r), 
                                 block=blk, grid=grd, shared=shared, stream=self.queue)
            
        # TODO: is this threshold acceptable in all cases?
        if (stdx <= 0.1 and stdy <= 0.1):
            output = input

        return output