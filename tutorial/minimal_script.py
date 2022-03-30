# This tutorial explains the minimal settings to get a reconstruction
# runnig in |ptypy|_. A |ptypy| script consists of two parts:

# * Creation of a parameter tree with parameters
#   as listed in :ref:`parameters` and
# * calling a :any:`Ptycho` instance with this parameter tree,
#   specifying a level that determines how much the Ptycho
#   instance will do.

# Preparing the parameter tree
# ----------------------------

# We begin with opening an empty python file of arbitrary name
# in an editor of your choice, e.g.::
#
#    $ gedit minimal_script.py

# Next we create an empty parameter tree. In |ptypy|, parameters
# are managed by the :any:`Param` class which is a convenience class subclassing Python's
# dict type. It is designed in such a way that dictionary items can be accessed also as
# class attributes, making the scripts and code much more readable.
from ptypy import utils as u
p = u.Param()  # root level

# We set the verbosity to a high level, in order to have information on the
# reconstruction process printed to the terminal.
# See :py:data:`~ptycho.verbose_level`.
p.verbose_level = "info"

# We limit this reconstruction to single precision. The other choice is to
# use double precision.
p.data_type = "single"

# We give this reconstruction the name ``'minimal'`` although it
# will automatically choose one from the file name of the script if we put in 
# ``None``. (But then the tutorial may not work on your computer as the chosen
# run name may differ from the one that this tutorial was created with)
p.run = 'minimal'

# Next, we set the home path. The :any:`Ptycho` instance will use this
# path as base for any other relative file path (e.g :py:data:`.io.autosave.path`
# or :py:data:`.io.rfile`). Relative paths lack a leading "/"
# (or "C:\\" on windows). Make sure to replace all "/" with "\\"
# if you run the scripts on a Windows system.
p.io = u.Param()
p.io.home = "/tmp/ptypy/"

# We want an intermediate result of the reconstruction
# to be dumped regularly every 20 reconstructions.
p.io.autosave = u.Param()
p.io.autosave.interval = 20

# In this tutorial we switch off the threaded plotting client.
# (alternative one-liners would be `p['io.autoplot.active'] = False`
# or `p.io['autoplot.active'] = False`)
p.io.autoplot = u.Param()
p.io.autoplot.active = False

# Since we do not want to plot anything, we don't need the
# interaction server either.
p.io.interaction = u.Param()
p.io.interaction.active = False

# Now we have to insert actual parameters associated with a
# ptychographic scan.

# PtyPy is designed to support reconstruction from mutliple scans.
# Each individual scan is represented by a branch in ``scans``. The parameters
# in these branches are those that differ from the *defaults* in the ``scan``
# branch mentioned above.
# Obviously at least the ``data`` branch will differ from
# :py:data:`.scan.data`. In this tutorial we
# create a new scan parameter branch ``MF``.
p.scans = u.Param()
p.scans.MF = u.Param()
p.scans.MF.name = 'Vanilla' 
p.scans.MF.data = u.Param()

# As data source we have choosen the *'MoonFlowerScan'* test source.
# That will make |ptypy| use the internal
# :py:class:`~ptypy.core.data.MoonFlowerScan` class to generate data.
# This class is meant for testing, and it provides/simulates
# diffraction patterns without using the more complex generic
# :py:class:`SimScan` class.
p.scans.MF.data.name = 'MoonFlowerScan'

# We set the diffraction frame shape to a small value (128x128px) and
# limit the number af diffraction patterns at 100. The
# :py:class:`~ptypy.core.data.MoonFlowerScan` instance will balance the
# diffraction patterns accordingly.
p.scans.MF.data.shape = 128
p.scans.MF.data.num_frames = 100

# We skip saving the "prepared" data file for now. The |ptypy| data
# management is described in detail in :ref:`ptypy_data`
p.scans.MF.data.save = None

# Needlees to say, we need to specify a reconstruction engine. We choose
# 40 iterations of difference map algorithm.
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 40
p.engines.engine00.numiter_contiguous = 5

# Running ptypy
# -------------

# We import the :any:`Ptycho` class and pass the tree ``p`` at level 5.
# This level tells :any:`Ptycho` to initialize everything and start the reconstruction
# using all reconstruction engines in ``p.engines`` immediately upon construction.
from ptypy.core import Ptycho
P = Ptycho(p, level=5)

# From the terminal log, we note that there was an autosave every 20
# iterations and the error reduced from iteration to iteration.
