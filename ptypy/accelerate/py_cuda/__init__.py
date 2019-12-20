import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import os
# debug_options = []
# debug_options = ['-O0', '-G', '-g', '-std=c++11', '--keep']
debug_options = ['-O3', '-DNDEBUG', '-std=c++11', '-lineinfo'] # release mode flags

context = None
queue = None

def get_context(new_queue=False):

    from ptypy.utils import parallel

    global context
    global queue

    if context is None:
        cuda.init()
        if parallel.rank_local < cuda.Device.count():
            import pyopencl as cl
            context = cuda.Device(parallel.rank_local).make_context()
            context.push()
        # print("made context %s on rank %s" % (str(context), str(parallel.rank)))
        # print("The cuda device count on %s is:%s" % (str(parallel.rank),
        #                                              str(cuda.Device.count())))
        # print("parallel.rank:%s, parallel.rank_local:%s" % (str(parallel.rank),
        #                                                     str(parallel.rank_local)))
    if queue is None:
        queue = cuda.Stream()
    return context, queue


def load_kernel(name, subs={}):

    fn = "%s/cuda/%s.cu" % (os.path.dirname(__file__), name)
    with open(fn, 'r') as f:
        kernel = f.read()
    for k,v in list(subs.items()):
        kernel = kernel.replace(k, str(v))
    mod = SourceModule(kernel, include_dirs=[np.get_include()], no_extern_c=True, options=debug_options)
    return mod.get_function(name)

