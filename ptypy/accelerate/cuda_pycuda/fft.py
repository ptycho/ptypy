from pycuda.compiler import SourceModule

import numpy as np


class FFT(object):

    def __init__(self, array, queue=None,
                 inplace=False,
                 pre_fft=None,
                 post_fft=None,
                 symmetric=True,
                 forward=True):
        """
        array should be gpuarray already
        """
        self._queue = queue
        from pycuda import gpuarray
        ## reikna
        from reikna import cluda
        api = cluda.cuda_api()
        thr = api.Thread(queue)

        dims = array.ndim
        if dims < 2:
            raise AssertionError('Input array must be at least 2-dimensional')
        axes = (array.ndim - 2, array.ndim - 1)

        # build the fft
        from reikna.fft import fft
        ftreikna = fft.FFT(array, axes)

        # attach scaling
        from reikna.transformations import mul_param
        sc = mul_param(array, np.float32)
        ftreikna.parameter.output.connect(sc, sc.input, out=sc.output, scale=sc.param)
        iscale = np.sqrt(np.prod(array.shape[-2:])) if symmetric else 1.0
        scale = 1.0 / iscale

        # attach arbitrary multiplication
        from reikna import core as rc
        from reikna.cluda import functions
        # get the IO type
        T_io = ftreikna.parameter[0]
        T_2d = rc.Type(T_io.dtype, T_io.shape[-2:])
        tr = rc.Transformation(
            [
                rc.Parameter('output', rc.Annotation(T_io, 'o')),
                rc.Parameter('fac', rc.Annotation(T_2d, 'i')),
                rc.Parameter('input', rc.Annotation(T_io, 'i')),
            ],
            """
            const VSIZE_T x = ${idxs[%d]};
            const VSIZE_T y = ${idxs[%d]};

            ${output.store_same}(${mul}(${input.load_same}, ${fac.load_idx}(x,y)));
            """ % axes,
            render_kwds={'mul': functions.mul(T_io.dtype, T_io.dtype)},
            connectors=['input', 'output']
        )

        if pre_fft is None and post_fft is None:
            self.pre_fft = self.post_fft = None
        elif pre_fft is not None and post_fft is None:
            self.pre_fft = thr.to_device(pre_fft)
            self.post_fft = None
            ftreikna.parameter.input.connect(tr, tr.output, pre_fft=tr.fac, data=tr.input)
        elif pre_fft is None and post_fft is not None:
            self.post_fft = thr.to_device(post_fft)
            self.pre_fft = None
            ftreikna.parameter.out.connect(tr, tr.input, post_fft=tr.fac, result=tr.output)
        else:
            self.pre_fft = thr.to_device(pre_fft)
            self.post_fft = thr.to_device(post_fft)
            ftreikna.parameter.input.connect(tr, tr.output, pre_fft=tr.fac, data=tr.input)
            ftreikna.parameter.out.connect(tr, tr.input, post_fft=tr.fac, result=tr.output)
        
        self._ftreikna_raw = ftreikna
        self._scale = scale
        self._iscale = iscale
        self._set_stream(thr)

    @property
    def queue(self):
        return self._queue

    @queue.setter
    def queue(self, stream):
        self._queue = stream
        from reikna import cluda
        api = cluda.cuda_api()
        thr = api.Thread(stream)
        self._set_stream(thr)

    def _set_stream(self, thr):
        self._ftreikna = self._ftreikna_raw.compile(thr)

        if self.pre_fft is None and self.post_fft is None:
            self.ft = lambda x, y: self._ftreikna(y, self._scale, x, 0)
            self.ift = lambda x, y: self._ftreikna(y, self._iscale, x, 1)
        elif self.pre_fft is not None and self.post_fft is None:
            self.ft = lambda x, y: self._ftreikna(y, self._scale, self.pre_fft, x, 0)
            self.ift = lambda x, y: self._ftreikna(y, self._iscale, self.pre_fft, x, 1)
        elif self.pre_fft is None and self.post_fft is not None:
            self.ft = lambda x, y: self._ftreikna(y, self.post_fft, self._scale, x, 0)
            self.ift = lambda x, y: self._ftreikna(y, self.post_fft, self._iscale, x, 1)
        else:
            self.ft = lambda x, y: self._ftreikna(y, self.post_fft, self._scale, self.pre_fft, x, 0)
            self.ift = lambda x, y: self._ftreikna(y, self.post_fft, self._iscale, self.pre_fft, x, 1)


    # self.queue = queue
    # apply_filter_code = """
    # #include <iostream>
    # #include <utility>
    # #include <thrust/complex.h>
    # #include <stdio.h>
    # using thrust::complex;
    #
    # extern "C"{
    #     __global__ void apply_filter(complex<float> *data,
    #                                     const complex<float> *__restrict__ filter,
    #                                     float sc,
    #                                     int batchsize,
    #                                     int size)
    #     {
    #       int offset = threadIdx.x + blockIdx.x * blockDim.x;
    #       int total = batchsize * size;
    #       if (offset >= total)
    #         return;
    #       complex<float> val = data[offset];
    #       if (filter)
    #       {
    #         val = filter[offset % size] * val;
    #       }
    #       val *= sc;
    #       data[offset] = val;
    #     }
    # }
    # """
    # self.apply_filter = SourceModule(apply_filter_code, include_dirs=[np.get_include()],
    #                                  no_extern_c=True).get_function("apply_filter")
    # sc = isc = 1.0 / np.sqrt(array.shape[-2:])
    #
    # plan = cu_fft.Plan(array.shape[-2:], np.complex64, np.complex64, array.shape[0])
    # empty_filter = np.ones((array.shape[-2:])) + 1j * np.ones((array.shape[-2:]))
    # if pre_fft is None and post_fft is None:
    #     pre_fft = gpuarray.to_gpu(empty_filter)
    #     post_fft = gpuarray.to_gpu(empty_filter)
    # elif pre_fft is not None and post_fft is None:
    #     post_fft = gpuarray.to_gpu(empty_filter)
    # elif pre_fft is None and post_fft is not None:
    #     pre_fft = gpuarray.to_gpu(empty_filter)
    #
    # batch_size = np.int32(array.shape[0])
    # block = 256
    # total = np.int32(np.prod(array.shape))
    # blocks = int((total + block - 1) // block)
    #
    # def ft(array, out_array):
    #     self.apply_filter(array, pre_fft, np.float32(1.0), batch_size, np.int32(np.prod(array.shape[-2:])),
    #                       block=(block, 1, 1),
    #                       grid=(blocks, 1, 1),)
    #     cu_fft.fft(array, out_array, plan)
    #     self.apply_filter(out_array, post_fft, np.float32(sc), batch_size, np.int32(np.prod(array.shape[-2:])),
    #                       block=(block, 1, 1),
    #                       grid=(blocks, 1, 1))
    # def ift(array, out_array):
    #     self.apply_filter(array, pre_fft, np.float32(1.0), batch_size, np.int32(np.prod(array.shape[-2:])),
    #                       block=(block, 1, 1),
    #                       grid=(blocks, 1, 1))
    #     print("here")
    #     cu_fft.fft(array, out_array, plan, True)
    #     print("here now")
    #     self.apply_filter(out_array, post_fft, np.float32(isc), batch_size, np.int32(np.prod(array.shape[-2:])),
    #                       block=(block, 1, 1),
    #                       grid=(blocks, 1, 1))
    #     print("done")
    #
    # self.ft = ft
    # self.ift = ift
    # print("Setup the fft")
