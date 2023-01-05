import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import os
kernel_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cuda_common'))
debug_options = ['-O3', '-DNDEBUG', '-lineinfo', '-I' + kernel_dir] # release mode flags

# C++14 support was added with CUDA 9, so we only enable the flag there
if cuda.get_version()[0] >= 9:
    debug_options += ['-std=c++14']
else:
    debug_options += ['-std=c++11']

context = None
queue = None

def get_context(new_context=False, new_queue=False):

    from ptypy.utils import parallel

    global context
    global queue

    if context is None or new_context:
        cuda.init()
        if parallel.rank_local >= cuda.Device.count():
            raise Exception('Local rank must be smaller than total device count, \
                rank={}, rank_local={}, device_count={}'.format(
                parallel.rank, parallel.rank_local, cuda.Device.count()
            ))
        context = cuda.Device(parallel.rank_local).make_context()
        context.push()
        # print("made context %s on rank %s" % (str(context), str(parallel.rank)))
        # print("The cuda device count on %s is:%s" % (str(parallel.rank),
        #                                              str(cuda.Device.count())))
        # print("parallel.rank:%s, parallel.rank_local:%s" % (str(parallel.rank),
        #                                                     str(parallel.rank_local)))
    if queue is None or new_queue:
        queue = cuda.Stream()
    
    return context, queue


def load_kernel(name, subs={}, file=None):

    if file is None:
        if isinstance(name, str):
            fn = "%s/%s.cu" % (kernel_dir, name)
        else:
            raise ValueError("name parameter must be a string if not filename is given")
    else:
        fn = "%s/%s" % (kernel_dir, file)

    with open(fn, 'r') as f:
        kernel = f.read()
    for k,v in list(subs.items()):
        kernel = kernel.replace(k, str(v))
    # insert a preprocessor line directive to assist compiler errors
    escaped = fn.replace("\\", "\\\\")
    kernel = '#line 1 "{}"\n'.format(escaped) + kernel
    mod = SourceModule(kernel, include_dirs=[np.get_include()], no_extern_c=True, options=debug_options)
    
    if isinstance(name, str):
        return mod.get_function(name)
    else:  # tuple
        return tuple(mod.get_function(n) for n in name)

