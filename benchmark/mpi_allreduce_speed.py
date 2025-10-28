import numpy as np
from ptypy.utils import parallel
from mpi4py import MPI
import time

sizes ={
    'i08': (1, 960, 960),
    'i13': (1, 13408, 13408),  
    'i14_1': (1, 8160, 8160),  
    'i14_2': (1, 3360, 3360), 
}

def run_benchmark(shape):
    megabytes = np.prod(shape) * 8 / 1024 / 1024 * 2

    data = np.zeros(shape, dtype=np.complex64)
    
    # average 5 runs
    duration = 0
    for n in range(5):
        t1 = time.perf_counter()
        parallel.allreduce(data)  # 2 calls to simulate ptypy obb / obn reduce
        parallel.allreduce(data)
        t2 = time.perf_counter()
        duration += t2-t1
    duration /= 5

    total = parallel.allreduce(duration, MPI.MAX)

    return megabytes, duration

res = []

for name,sz in sizes.items():
    mb, dur = run_benchmark(sz)
    res.append([name, dur, mb, mb/dur])

if parallel.rank == 0:
    print('Final results for {} processes'.format(parallel.size))
    print(','.join(['Name', 'Duration', 'MB', 'MB/s']))
    for r in res:
        print(','.join([str(x) for x in r]))
