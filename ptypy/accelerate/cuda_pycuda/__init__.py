import sys
import os

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.tools import DeviceMemoryPool

from ptypy.utils import parallel

kernel_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cuda_common'))
debug_options = ['-O3', '-DNDEBUG', '-lineinfo', '-I' + kernel_dir] # release mode flags

# C++14 support was added with CUDA 9, so we only enable the flag there
if cuda.get_version()[0] >= 9:
    debug_options += ['-std=c++14']
else:
    debug_options += ['-std=c++11']

# ensure pycuda's make_default_context picks up the correct GPU for that rank
os.environ['CUDA_DEVICE'] = str(parallel.rank_local)

queue = None
dev_pool = None


def _pycuda_excepthook(type, value, tb):
    global dev_pool

    # memory pool clean-up, avoid memory leak in the case of raising exception
    if dev_pool is not None:
        # only do the clean-up if it is present
        dev_pool.stop_holding()

    # raise the original exception
    sys.__excepthook__(type, value, tb)
sys.excepthook = _pycuda_excepthook


def get_context(new_queue=False):

    global queue

    # idempotent anyway
    cuda.init()

    if parallel.rank_local >= cuda.Device.count():
        raise Exception('Local rank must be smaller than total device count, \
            rank={}, rank_local={}, device_count={}'.format(
            parallel.rank, parallel.rank_local, cuda.Device.count()
        ))

    # the existing context will always be the primary context, unless
    # explicitly created elsewhere
    if (context := cuda.Context.get_current()) is None:
        from pycuda import autoprimaryctx
        context = autoprimaryctx.context

    if queue is None or new_queue:
        queue = cuda.Stream()

    return context, queue


def get_dev_pool():
    global dev_pool

    # retain a single global instance of device memory pool
    if dev_pool is None:
        dev_pool = DeviceMemoryPool()

    return dev_pool


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
