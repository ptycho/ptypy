from . import load_kernel
from pycuda import gpuarray
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
