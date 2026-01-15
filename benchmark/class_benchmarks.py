import numpy as np
import ptypy
from ptypy import utils as u
from ptypy.core import View, Container, Storage, Base, POD
from memory_profiler import profile
import gc 
nviews = 5000
steps = 4


C1 = Container(data_type='real')
C2 = Container(data_type='real')
C3 = Container(data_type='real')
C4 = Container(data_type='real')
C5 = Container(data_type='real')
S1 = C1.new_storage(shape=(1, 7, 7))
S2 = C2.new_storage(shape=(1, 7, 7))
S3 = C3.new_storage(shape=(1, 7, 7))
S4 = C4.new_storage(shape=(1, 7, 7))
S5 = C5.new_storage(shape=(1, 7, 7))
B= Base()
V1 = View(C1, ID=None, storageID = S1.ID, psize = None, shape = (4, 4), coord = 0.,)
@profile
def add_views(nviews):
    for k in range(int(nviews)):
        V1 = View(C1, ID=None, storageID = S1.ID, psize = None, shape = (4, 4), coord = 0.,)
        V2 = View(C1, ID=None, storageID = S1.ID, psize = None, shape = (4, 4), coord = 0.,)
        V3 = View(C1, ID=None, storageID = S1.ID, psize = None, shape = (4, 4), coord = 0.,)
        V4 = View(C1, ID=None, storageID = S1.ID, psize = None, shape = (4, 4), coord = 0.,)
        V5 = View(C1, ID=None, storageID = S1.ID, psize = None, shape = (4, 4), coord = 0.,)
        POD(B,views = {'probe': V1, 'obj': V2, 'exit': V3, 'diff': V4, 'mask': V5})
        #B2 = Base(B,ID=None)
        #pass
    
for k in range(int(steps)):
    add_views(nviews)
    gc.collect()
    print(k) 
    print(list(C1._recs.values())[0].nbytes / 1e6)

print(C1.formatted_report())
u.pause(1)
gc.collect()
u.pause(4)
u.pause(4)
print(C1.formatted_report())
add_views(nviews)
u.pause(4)

"""
from guppy import hpy
h = hpy()
print h.heap() 
"""
