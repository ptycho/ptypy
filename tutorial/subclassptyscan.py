# In this tutorial, we learn how to subclass :py:class:`PtyScan` to make
# ptypy work with any experimental setup.

# This tutorial can be used as a direct follow-up to :ref:`simupod`
# if section :ref:`store` was completed

# Again, the imports first.
import numpy as np
from ptypy.core.data import PtyScan
from ptypy import utils as u

# For this tutorial we assume that the data and meta information is
# in this path:
save_path = '/tmp/ptypy/sim/'

# Furthermore, we assume that a file about the experimental geometry is
# located at
geofilepath = save_path + 'geometry.txt'
print(geofilepath)
# and has contents of the following form
print(''.join([line for line in open(geofilepath, 'r')]))

# The scanning positions are in
positionpath = save_path + 'positions.txt'
print(positionpath)

# with a list of positions for vertical and horizontanl movement and the
# image frame from the "camera"
print(''.join([line for line in open(positionpath, 'r')][:6])+'....')

# Writing a subclass
# ------------------

# The simplest subclass of PtyScan would look like this
class NumpyScan(PtyScan):
    """
    A PtyScan subclass to extract data from a numpy array.
    """

    def __init__(self, pars=None, **kwargs):
        # In init we need to call the parent.
        super(NumpyScan, self).__init__(pars, **kwargs)

# Of course this class does nothing special beyond PtyScan.
# As it is, the class also cannot be used as a real PtyScan instance
# because its defaults are not properly managed. For this, Ptypy provides a
# powerful self-documenting tool call a "descriptor" which can be applied
# to any new class using a decorator. The tree of all valid ptypy parameters
# is located at :ref:`here <parameters>`. To manage the default
# parameters of our subclass and document its existence, we would need to write
from ptypy import defaults_tree

@defaults_tree.parse_doc('scandata.numpyscan')
class NumpyScan(PtyScan):
    """
    A PtyScan subclass to extract data from a numpy array.
    """

    def __init__(self, pars=None, **kwargs):
        # In init we need to call the parent.
        super(NumpyScan, self).__init__(pars, **kwargs)

# The decorator extracts information from the docstring of the subclass and
# parent classes about the expected input parameters. Currently the docstring
# of `NumpyScan` does not contain anything special, thus the only parameters
# registered are those of the parent class, `PtyScan`:
print(defaults_tree['scandata.numpyscan'].to_string())

# As you can see, there are already many parameters documented in `PtyScan`'s
# class. For each parameter, most important are the *type*, *default* value and
# *help* string. The decorator does more than collect this information: it also
# generates from it a class variable called `DEFAULT`, which stores all defaults:
print(u.verbose.report(NumpyScan.DEFAULT, noheader=True))

# Now we are ready to add functionality to our subclass.
# A first step of initialisation would be to retrieve
# the geometric information that we stored in ``geofilepath`` and update
# the input parameters with it.

# We write a tiny file parser.
def extract_geo(base_path):
    out = {}
    with open(base_path+'geometry.txt') as f:
        for line in f:
            key, value = line.strip().split()
            out[key] = eval(value)
    return out

# We test it.
print(extract_geo(save_path))

# That seems to work. We can integrate this parser into
# the initialisation as we assume that this small access can be
# done by all MPI nodes without data access problems. Hence,
# our subclass becomes
@defaults_tree.parse_doc('scandata.numpyscan')
class NumpyScan(PtyScan):
    """
    A PtyScan subclass to extract data from a numpy array.

    Defaults:

    [name]
    type = str
    default = numpyscan
    help =

    [base_path]
    type = str
    default = './'
    help = Base path to extract data files from.
    """

    def __init__(self, pars=None, **kwargs):
        p = self.DEFAULT.copy(depth=2)
        p.update(pars)

        with open(p.base_path+'geometry.txt') as f:
            for line in f:
                key, value = line.strip().split()
                # we only replace Nones or missing keys
                if p.get(key) is None:
                    p[key] = eval(value)

        super(NumpyScan, self).__init__(p, **kwargs)

# We now need a new input parameter called `base_path`, so we documented it
# in the docstring after the section header "Defaults:".
print(defaults_tree['scandata.numpyscan.base_path'])

# As you can see, the first step in `__init__` is to build a default
# parameter structure to ensure that all input parameters are available.
# The next line updates this structure to overwrite the entries specified by
# the user.

# Good! Next, we need to implement how the class finds out about
# the positions in the scan. The method
# :py:meth:`~ptypy.core.data.PtyScan.load_positions` can be used
# for this purpose.
print(PtyScan.load_positions.__doc__)

# The parser for the positions file would look like this.
def extract_pos(base_path):
    pos = []
    files = []
    with open(base_path+'positions.txt') as f:
        for line in f:
            fname, y, x = line.strip().split()
            pos.append((eval(y), eval(x)))
            files.append(fname)
    return files, pos

# And the test:
files, pos = extract_pos(save_path)
print(files[:2])
print(pos[:2])

@defaults_tree.parse_doc('scandata.numpyscan')
class NumpyScan(PtyScan):
    """
    A PtyScan subclass to extract data from a numpy array.

    Defaults:

    [name]
    type = str
    default = numpyscan
    help =

    [base_path]
    type = str
    default = /tmp/ptypy/sim/
    help = Base path to extract data files from.
    """

    def __init__(self, pars=None, **kwargs):
        p = self.DEFAULT.copy(depth=2)
        p.update(pars)

        with open(p.base_path+'geometry.txt') as f:
            for line in f:
                key, value = line.strip().split()
                # we only replace Nones or missing keys
                if p.get(key) is None:
                    p[key] = eval(value)

        super(NumpyScan, self).__init__(p, **kwargs)

    def load_positions(self):
        # the base path is now stored in
        base_path = self.info.base_path
        pos = []
        with open(base_path+'positions.txt') as f:
            for line in f:
                fname, y, x = line.strip().split()
                pos.append((eval(y), eval(x)))
                files.append(fname)
        return np.asarray(pos)

# One nice thing about rewriting ``self.load_positions`` is that
# the maximum number of frames will be set and we do not need to
# manually adapt :py:meth:`~ptypy.core.data.PtyScan.check`

# The last step is to overwrite the actual loading of data.
# Loading happens (MPI-compatible) in
# :py:meth:`~ptypy.core.data.PtyScan.load`
print(PtyScan.load.__doc__)

# Load seems a bit more complex than ``self.load_positions`` for its
# return values. However, we can opt-out of providing weights (masks)
# and positions, as we have already adapted ``self.load_positions``
# and there were no bad pixels in the (linear) detector

# The final subclass looks like this. We overwrite two defaults from
# `PtyScan`:
@defaults_tree.parse_doc('scandata.numpyscan')
class NumpyScan(PtyScan):
    """
    A PtyScan subclass to extract data from a numpy array.

    Defaults:

    [name]
    type = str
    default = numpyscan
    help =

    [base_path]
    type = str
    default = /tmp/ptypy/sim/
    help = Base path to extract data files from.

    [auto_center]
    default = False

    [dfile]
    default = /tmp/ptypy/sim/npy.ptyd
    """

    def __init__(self, pars=None, **kwargs):
        p = self.DEFAULT.copy(depth=2)
        p.update(pars)

        with open(p.base_path+'geometry.txt') as f:
            for line in f:
                key, value = line.strip().split()
                # we only replace Nones or missing keys
                if p.get(key) is None:
                    p[key] = eval(value)

        super(NumpyScan, self).__init__(p, **kwargs)

    def load_positions(self):
        # the base path is now stored in
        base_path = self.info.base_path
        pos = []
        with open(base_path+'positions.txt') as f:
            for line in f:
                fname, y, x = line.strip().split()
                pos.append((eval(y), eval(x)))
                files.append(fname)
        return np.asarray(pos)

    def load(self, indices):
        raw = {}
        bp = self.info.base_path
        for ii in indices:
            raw[ii] = np.load(bp+'ccd/diffraction_%04d.npy' % ii)
        return raw, {}, {}

# Loading the data
# ----------------

# With the subclass we create a scan only using defaults
NPS = NumpyScan()
NPS.initialize()

# In order to process the data. We need to call
# :py:meth:`~ptypy.core.data.PtyScan.auto` with the chunk size
# as arguments. It returns a data chunk that we can inspect
# with :py:func:`ptypy.utils.verbose.report`. The information is
# concatenated, but the length of iterables or dicts is always indicated
# in parantheses.
print(u.verbose.report(NPS.auto(80), noheader=True))
print(u.verbose.report(NPS.auto(80), noheader=True))

# We observe the second chunk was not 80 frames deep but 34
# as we only had 114 frames of data.

# So where is the *.ptyd* data-file? As default, PtyScan does not
# actually save data. We have to manually activate it in in the
# input paramaters.
data = NPS.DEFAULT.copy(depth=2)
data.save = 'append'
NPS = NumpyScan(pars=data)
NPS.initialize()
for i in range(50):
    msg = NPS.auto(20)
    if msg == NPS.EOS:
        break

# We can analyse the saved ``npy.ptyd`` with
# :py:func:`~ptypy.io.h5IO.h5info`
from ptypy.io import h5info
print(h5info(NPS.info.dfile))

