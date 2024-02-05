import pyopencl as cl
from pyopencl import array as cla
import numpy as np
import time


class FFT_2D_ocl_gpyfft(object):

    def __init__(self, queue, array,
                 pre_fft=None,
                 post_fft=None,
                 inplace=False,
                 symmetric=False):

        import gpyfft
        GFFT = gpyfft.GpyFFT(debug=False)
        import gpyfft.gpyfftlib as gfft

        self.queue = queue

        a = np.empty_like(array)

        dims = array.ndim

        if dims == 2:
            a = a.reshape((1,) + a.shape)
        elif dims == 3:
            pass
        else:
            raise AssertionError('Input array must be 2 or 3-dimensional')

        shape = array.shape[-2:]
        distance = a.strides[0] / a.itemsize
        strides = (a.strides[1] / a.itemsize, a.strides[2] / a.itemsize)

        batchsize = a.shape[0]

        if a.dtype.type is np.complex64:
            precision = gfft.CLFFT_SINGLE
        elif a.dtype.type is np.complex128:
            precision = gfft.CLFFT_DOUBLE
        else:
            raise AssertionError(
                'Input array data type must be either complex64 or complex128 but is %s' % str(a.dtype.name))

        layout = gfft.CLFFT_COMPLEX_INTERLEAVED

        plan = GFFT.create_plan(self.queue.context, shape)
        plan.inplace = inplace
        plan.strides_in = strides
        plan.strides_out = strides
        plan.distances = (distance, distance)
        plan.batch_size = batchsize
        plan.precision = precision
        plan.layouts = (layout, layout)
        if symmetric:
            plan.scale_forward /= np.sqrt(np.prod(shape))
            plan.scale_backward *= np.sqrt(np.prod(shape))
        self.plan = plan
        # self.print_plan_info()
        # allocate buffers
        CA = cl.tools.ImmediateAllocator(self.queue, cl.mem_flags.READ_ONLY)

        if pre_fft is not None:
            if np.isscalar(pre_fft):
                pre_fft = np.ones(plan.shape, a.dtype) * pre_fft

            self.pre_fft = cla.to_device(queue, pre_fft)  # allocator = CA)
            queue.finish()

            precallbackstr = "#define PLANE %d\n" % np.prod(plan.shape)
            precallbackstr += """float2 prefft(__global void* input, \n
                                  uint inoffset,                      \n
                                  __global void* userdata)             \n
                                {                                                            \n
                                float2 fac = *((__global float2*)userdata + inoffset % PLANE);      \n 
                                float2 in = *((__global float2*)input + inoffset); \n
                                float2 ret; \n 
                                ret.x = in.x * fac.x - in.y * fac.y ; \n
                                ret.y = in.x * fac.y + in.y * fac.x ; \n
                                return ret;         \n                                         
                                }\n"""
            if precision is gfft.CLFFT_DOUBLE:
                precallbackstr = precallbackstr.replace('float2', 'double2')
            plan.set_callback("prefft", precallbackstr, 'pre', user_data=self.pre_fft.data)

        if post_fft is not None:
            if np.isscalar(post_fft):
                post_fft = np.ones(plan.shape, a.dtype) * post_fft

            # A = np.ones(self.t_shape, out_array.dtype)
            self.post_fft = cla.to_device(queue, post_fft)  # allocator = CA)
            queue.finish()
            postcallbackstr = "#define PLANE %d\n" % np.prod(plan.shape)
            postcallbackstr += """void postfft(__global void* output, \n
                                  uint outoffset,                     \n
                                  __global void* userdata,            \n
                                float2 fftoutput)            \n
                                {                                                   \n
                                float2 fac = *((__global float2*)userdata + outoffset % PLANE);    \n
                                float2 res; \n
                                res.x = fftoutput.x * fac.x - fftoutput.y * fac.y;\n
                                res.y = fftoutput.x * fac.y + fftoutput.y * fac.x;\n
                                *((__global float2*)output + outoffset) = res;\n
                                }\n"""
            if precision is gfft.CLFFT_DOUBLE:
                postcallbackstr = postcallbackstr.replace('float2', 'double2')
            plan.set_callback("postfft", postcallbackstr, 'post', user_data=self.post_fft.data)

        plan.bake(self.queue)
        temp_size = plan.temp_array_size
        if temp_size:
            self.temp_buffer = cl.Buffer(self.queue.context, cl.mem_flags.READ_WRITE, size=temp_size)
        else:
            self.temp_buffer = None

        self.plan = plan

    def _ft(self, inarray, outarray=None, forward=True):
        if not self.plan.inplace and outarray is None:
            raise RuntimeError('Specify an opencl array to store the results')

        elif self.plan.inplace:
            events = self.plan.enqueue_transform((self.queue,), (inarray.data,),
                                                 direction_forward=forward, temp_buffer=self.temp_buffer)
        else:
            events = self.plan.enqueue_transform((self.queue,), (inarray.data,), (outarray.data,),
                                                 direction_forward=forward, temp_buffer=self.temp_buffer)

        return events

    def ft(self, inarray, outarray=None):

        return self._ft(inarray, outarray, True)

    def ift(self, inarray, outarray=None):

        return self._ft(inarray, outarray, False)

    # def print_plan_info(self):
    #     plan = self.plan
    #     print('in_array.shape:          ', plan.shape)
    #     print('in_array.strides/itemsize', tuple(s // in_array.dtype.itemsize for s in in_array.strides))
    #     print('shape transform          ', t_shape)
    #     print('t_strides                ', t_strides_in)
    #     print('distance_in              ', t_distance_in)
    #     print('batchsize                ', t_batchsize_in)
    #     print('t_stride_out             ', t_strides_out)
    #     print('inplace                  ', t_inplace)


class FFT_2D_ocl_reikna(object):

    def __init__(self, queue, array,
                 pre_fft=None,
                 post_fft=None,
                 inplace=False,
                 symmetric=True):

        self.queue = queue
        ## reikna
        from reikna import cluda
        api = cluda.ocl_api()
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
        sc = mul_param(array, float)
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
            self._ftreikna = ftreikna.compile(thr)
            self.ft = lambda x, y: self._ftreikna(y, scale, x, 0)
            self.ift = lambda x, y: self._ftreikna(y, iscale, x, 1)

        elif pre_fft is not None and post_fft is None:
            self.pre_fft = cla.to_device(queue, pre_fft)
            ftreikna.parameter.input.connect(tr, tr.output, pre_fft=tr.fac, data=tr.input)
            self._ftreikna = ftreikna.compile(thr)
            self.ft = lambda x, y: self._ftreikna(y, scale, self.pre_fft, x, 0)
            self.ift = lambda x, y: self._ftreikna(y, iscale, self.pre_fft, x, 1)

        elif pre_fft is None and post_fft is not None:
            self.post_fft = cla.to_device(queue, post_fft)
            ftreikna.parameter.out.connect(tr, tr.input, post_fft=tr.fac, result=tr.output)
            self._ftreikna = ftreikna.compile(thr)
            self.ft = lambda x, y: self._ftreikna(y, self.post_fft, scale, x, 0)
            self.ift = lambda x, y: self._ftreikna(y, self.post_fft, iscale, x, 1)

        else:
            self.pre_fft = cla.to_device(queue, pre_fft)
            self.post_fft = cla.to_device(queue, post_fft)
            ftreikna.parameter.input.connect(tr, tr.output, pre_fft=tr.fac, data=tr.input)
            ftreikna.parameter.out.connect(tr, tr.input, post_fft=tr.fac, result=tr.output)
            # print self._ftreikna.signature.parameters.keys()
            self._ftreikna = ftreikna.compile(thr)
            self.ft = lambda x, y: self._ftreikna(y, self.post_fft, scale, self.pre_fft, x, 0)
            self.ift = lambda x, y: self._ftreikna(y, self.post_fft, iscale, self.pre_fft, x, 1)

        queue.finish()
