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
# For sure we set.
DEFAULT.recipe = RECIPE

# A default data file location may be handy too and we allow saving of
# data in a single file. And since we now it is simulated dat we do not
# have to find the optical axes in the diffraction pattern with
# the help of auto_center
DEFAULT.dfile = '/tmp/ptypy/sim/npy.ptyd'
DEFAULT.auto_center = False

# Our defaults are now
print u.verbose.report(DEFAULT,noheader=True)

# The simplest subclass of PtyScan would look like this
class NumpyScan(PtyScan):
    # We overwrite the DEFAULT with the new DEFAULT.
    DEFAULT = DEFAULT
    
    def __init__(self,pars=None, **kwargs):
        # In init we need to call the parent.
        super(NumpyScan, self).__init__(pars, **kwargs)

# Of course this class does nothing special beyond PtyScan.

# An additional step of initialisation would be to retrieve 
# the geometric information that we stored in ``geofilepath`` and update
# the input parameters with it.
 
# We write a tiny file parser.
def extract_geo(base_path):
    out = {}
    with open(base_path+'geometry.txt') as f:
        for line in f:
            key, value = line.strip().split()
            out[key]=eval(value)
    return out

# We test it.
print extract_geo(save_path)

# That seems to work. We can integrate this parser into 
# the initialisation as we assume that this small access can be 
# done by all MPI nodes without data access problems. Hence,
# our subclass becomes
class NumpyScan(PtyScan):
    # We overwrite the DEFAULT with the new DEFAULT.
    DEFAULT = DEFAULT
    
    def __init__(self,pars=None, **kwargs):
        p = DEFAULT.copy(depth=2)
        p.update(pars) 
        
        with open(p.recipe.base_path+'geometry.txt') as f:
            for line in f:
                key, value = line.strip().split()
                # we only replace Nones or missing keys
                if p.get(key) is None:
                    p[key]=eval(value)
        
        super(NumpyScan, self).__init__(p, **kwargs)

# Good! Next, we need to implement how the class finds out about
# the positions in the scan. The method 
# :py:meth:`~ptypy.core.data.PtyScan.load_positions` can be used
# for this purpose.
print PtyScan.load_positions.__doc__

# The parser for the positions file would look like this.
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

class NumpyScan(PtyScan):
    # We overwrite the DEFAULT with the new DEFAULT.
    DEFAULT = DEFAULT
    
    def __init__(self,pars=None, **kwargs):
        p = DEFAULT.copy(depth=2)
        p.update(pars) 
        
        with open(p.recipe.base_path+'geometry.txt') as f:
            for line in f:
                key, value = line.strip().split()
                # we only replace Nones or missing keys
                if p.get(key) is None:
                    p[key]=eval(value)
        
        super(NumpyScan, self).__init__(p, **kwargs)
        # all input data is now in self.info
        
    def load_positions(self):
        # the base path is now stored in 
        base_path = self.info.recipe.base_path
        with open(base_path+'positions.txt') as f:
            for line in f:
                fname, y, x = line.strip().split()
                pos.append((eval(y),eval(x)))
                files.append(fname)
        return np.asarray(pos)

# One nice thing about rewriting ``self.load_positions`` is that the 
# the maximum number of frames will be set and we do not need to
# manually adapt :py:meth:`~ptypy.core.data.PtyScan.check`

# The last step is to overwrite the actual loading of data.
# Loading happens (MPI-compatible) in 
# :py:meth:`~ptypy.core.data.PtyScan.load`
print PtyScan.load.__doc__

# Load seems a bit more complex than ``self.load_positions`` for its 
# return values. However, we can opt-out of providing weights (masks)
# and positions, as we have already adapted ``self.load_positions``
# and we there were no bad pixels in the (linear) detector

# The final subclass looks like this.
class NumpyScan(PtyScan):
    # We overwrite the DEFAULT with the new DEFAULT.
    DEFAULT = DEFAULT
    
    def __init__(self,pars=None, **kwargs):
        p = DEFAULT.copy(depth=2)
        p.update(pars) 
        
        with open(p.recipe.base_path+'geometry.txt') as f:
            for line in f:
                key, value = line.strip().split()
                # we only replace Nones or missing keys
                if p.get(key) is None:
                    p[key]=eval(value)
        
        super(NumpyScan, self).__init__(p, **kwargs)
        # all input data is now in self.info
        
    def load_positions(self):
        # the base path is now stored in
        pos=[] 
        base_path = self.info.recipe.base_path
        with open(base_path+'positions.txt') as f:
            for line in f:
                fname, y, x = line.strip().split()
                pos.append((eval(y),eval(x)))
                files.append(fname)
        return np.asarray(pos)
    
    def load(self,indices):
        raw = {}
        bp = self.info.recipe.base_path
        for ii in indices:
            raw[ii] = np.load(bp+'ccd/diffraction_%04d.npy' % ii )
        return raw, {},{}

# Loading the data
# ----------------

# With the subclass we create a scan only using defaults
NPS = NumpyScan()
NPS.initialize()

# In order to process the data. We need to call 
# :py:meth:`~ptypy.core.data.PtyScan.auto` with the chunk size
# as arguments. It returns a data chunk that we can inspect
# with :py:func:`ptypy.utils.verbose.report`. The information is 
# concetanated, but the length of iterables or dicts is always indicated
# in parantheses.
print u.verbose.report(NPS.auto(80),noheader=True)
print u.verbose.report(NPS.auto(80),noheader=True)

# We observe the the second chunk was not 80 frames deep but 34
# as we only had 114 frames of data.

# So where is the *.ptyd* data-file? As default, PtyScan does not
# actually save data. We have to manually activate it in in the 
# input paramaters.
data = NPS.DEFAULT.copy(depth=2)
data.save = 'append'
NPS = NumpyScan(pars = data)
NPS.initialize()
for i in range(50):
    msg = NPS.auto(20)
    if msg==NPS.EOS:
        break

# We can analyse the saved ``npy.ptyd`` with 
# :py:func:`~ptypy.io.h5IO.h5info`
from ptypy.io import h5info
print h5info(NPS.info.dfile)

