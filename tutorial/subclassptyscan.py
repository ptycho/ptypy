# .. _subclassptyscan:

# Tutorial : Subclassing PtyScan
# ==============================

# In this tutorial, we learn how to subclass :any:`PtyScan` to make 
# ptypy work with any experimental setup.

# This tutorial can be used as a direct follow-up to :ref:`simupod` 
# if section :ref:`store` was completed

# Again, the imports first.
import matplotlib as mpl
import numpy as np
import ptypy
from ptypy import utils as u
plt = mpl.pyplot
import sys

# For this tutorial we assume, that the data and meta information is 
# in this path:
save_path = '/tmp/ptypy/sim/'

# Furthermore, we assume that a file about the experimental geometry is
# located at 
geofilepath = save_path+ 'geometry.txt'
print geofilepath
# and has contents of the following form
print ''.join([line for line in open(geofilepath,'r')])

# The scanning positions are in 
positionpath = save_path+ 'positions.txt'
print positionpath

# with a list of positions for vertical and horizontanl movement and the
# image frame from the "camera" 
print ''.join([line for line in open(positionpath,'r')][:6])+'....'

# Writing a subclass
# ------------------

# A subclass of :any:`PtyScan` takes the same input parameter 
# tree as PtyScan itself, i.e :py:data:`.scan.data`. As the subclass
# will most certainly require additional parameters, there has to be 
# a flexible additional container. For PtyScan, that is the 
# :py:data:`.scan.data.recipe` parameter. A subclass must extract all 
# additional parameters from this source and, in script, you fill
# the recipe with the appropriate items.

# In this case we can assume that the only parameter of the recipe
# is the base path ``/tmp/ptypy/sim/``\ . Hence we write
RECIPE = u.Param()
RECIPE.base_path = '/tmp/ptypy/sim/'

# Now we import the deafult generic parameter set from
from ptypy.core.data import PtyScan
DEFAULT = PtyScan.DEFAULT.copy()

# This would be the perfect point to change any default value.
# For sure we set
DEFAULT.recipe = RECIPE

class NumpyScan(PtyScan):
    # We overwrite the DEFAULT with the new DEFAULT.
    DEFAULT = DEFAULT
    # In init we need to call the parent.
    def __init__(self,pars=None, **kwargs):
        super(NumpyScan, self).__init__(p, **kwargs)

# At this point of initialisation it would be good to read in
# the geometric information we stored in ``geofilepath``. We write a 
# tiny file parser.
def extract_geo(base_path):
    out = {}
    with open(base_path+'geometry.txt') as f:
        for line in f:
            key, value = line.strip().split()
            out[key]=eval(value)
    return out

# We test it.
print extract_geo(save_path)

# Similarly we would need the same for the positions file
def extract_pos(base_path):
    pos = []
    files =[]
    with open(base_path+'positions.txt') as f:
        for line in f:
            fname, y, x = line.strip().split()
            pos.append((eval(y),eval(x)))
            files.append(fname)
    return files,pos

# And the test:
files, pos = extract_pos(save_path)
print files[:2]
print pos[:2]
