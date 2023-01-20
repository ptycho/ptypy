
from typing import Optional
import cupy as cp
import os

from ptypy.utils.verbose import headerline, log

kernel_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cuda_common'))
compile_options =['-std=c++14', '-DPTYPY_CUPY_NVTRC=1', '-I' + kernel_dir, '-DNDEBUG']
queue = None
device = None


def get_context(new_queue=False):

    from ptypy.utils import parallel

    global queue, device

    if queue is None or new_queue:
        ndevs = cp.cuda.runtime.getDeviceCount()
        if parallel.rank_local >= ndevs:
            raise Exception('Local rank must be smaller than total device count, \
                rank={}, rank_local={}, device_count={}'.format(
                parallel.rank, parallel.rank_local, ndevs
            ))
        device = cp.cuda.Device(parallel.rank_local)
        device.use()
        queue = cp.cuda.Stream()

    return queue


def load_kernel(name, subs={}, file=None, options=None):

    if file is None:
        if isinstance(name, str):
            fn = "%s/%s.cu" % (kernel_dir, name)
        else:
            raise ValueError(
                "name parameter must be a string if not filename is given")
    else:
        fn = "%s/%s" % (kernel_dir, file)

    with open(fn, 'r') as f:
        kernel = f.read()
    for k, v in list(subs.items()):
        kernel = kernel.replace(k, str(v))
    # insert a preprocessor line directive to assist compiler errors
    escaped = fn.replace("\\", "\\\\")
    kernel = '#line 1 "{}"\n'.format(escaped) + kernel

    opt = [*compile_options]
    if options is not None:
        opt += list(options)
    module = cp.RawModule(code=kernel, options=tuple(opt))
    if isinstance(name, str):
        return module.get_function(name)
    else:  # tuple
        return tuple(module.get_function(n) for n in name)

def log_device_memory_stats(level=4, heading: str ='Device Memory Stats'):
        mempool = cp.get_default_memory_pool()
        pinned_pool = cp.get_default_pinned_memory_pool()
        log(level, '\n' + headerline(heading))
        log(level, f'Device id             : {cp.cuda.Device().id}')
        log(level, f'Total device mem      : {cp.cuda.runtime.memGetInfo()[1]/1024/1024} MB')
        log(level, f'Free device mem       : {cp.cuda.runtime.memGetInfo()[0]/1024/1024} MB')
        log(level, f'MemoryPool size       : {mempool.total_bytes()/1024/1024} MB')
        log(level, f'MemoryPool used       : {mempool.used_bytes()/1024/1024} MB')
        log(level, f'MemoryPool limit      : {mempool.get_limit()/1024/1024} MB')
        log(level, f'MemoryPool free blocks: {mempool.n_free_blocks()}')
        log(level, f'PinnedPool free blocks: {pinned_pool.n_free_blocks()}')