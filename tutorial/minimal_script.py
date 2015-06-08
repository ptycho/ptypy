# This tutorial explains the minimal settings to get a reconstruction
# runnig in |ptypy|. A |ptypy| script consists of 2 parts, creation of
# a parameter tree with parameters as listed in :ref:`parameters` and 
# finally calling a :any:`Ptycho` instance with these parameter tree 
# and a certain level that determines how much the Ptycho 
# instance will do.

# Preparing the parameter tree
# ----------------------------

# We begin with creating an empty parameter tree. In |ptypy| parameters
# are managed by the :any:`Param` class which is a subclass of Pythons
# dict type and acts like a composite of 
# of dictionary and object: All items can be accessed as class attributes. 
from ptypy import utils as u
p = u.Param()  # root level

# We set the verbosity to a high level, such that we information on the
# reconstruction process are printed into the terminal. 
# See :py:data:`.verbose_level`.
p.verbose_level = 3                              

# We limit this reconstruction to single precision, but yoy may as well
# use double precision
p.data_type = "single"                           

# We give this reconstruction the name ``'minimal'`` although it 
# will automatically choose one from the script if we put in ``None``.
p.run = 'minimal'

# We set the home path. The reconstruction will save / dump anything
# into this base path if any other path in the tree lacks a leading "/" 
# (or "C:\" on windows). Make sure to replace all "/" with "\" if you run
# the scripts on a Windows system.
p.io = u.Param()
p.io.home = "/tmp/ptypy/"

# We want a state of the reconstruction to be dumped regularly every
# 10 reconstructions
p.io.autosave = u.Param()
p.io.autosave.interval = 10

# In this tutorial we switch of the threaded plotting
p.io.autoplot = False

# Since we are not plotting we do not need the interaction server either.
p.io.interaction = False

# This branch of the tree would hold all *common* parameters for scans. 
# If there is only one scan, it does not matter if we specify parameters
# for illumination or sample in this branch or in ``scans``. In this
# case we do not bother to enter paramter but leave the branch empty
# (That will fill it with the defaults :py:data:`.scan` ) 
p.scan = u.Param()

# The ``scans`` branch encloses all differences for a scan with respect
# to the ``scan`` branch mentioned above. For sure different for each 
# scan is the ``data`` branch :py:data:`.scan.data`. In this case we 
# create a new scan paramter branch ``MF``. We only specify the data
# branch and tell |ptypy| to use scan meta data when possible.
p.scans = u.Param()
p.scans.MF = u.Param()
p.scans.MF.if_conflict_use_meta = True
p.scans.MF.data= u.Param()

# As data source we choose the *'test'* source. This will make ptypy
# use the internal :py:class:`~ptypy.core.data.MoonFlowerScan` class.
# This class is a test class, that provides/ simulates diffraction 
# patterns without explicitly using the generic :any:`SimScan` class.
p.scans.MF.data.source = 'test'

# We set the diffraction frame shape to a small value and only allow 
# a hundred diffraction patterns at max. The 
# :py:class:`~ptypy.core.data.MoonFlowerScan` instance will balance the
# diffraction patterns accordingly.
p.scans.MF.data.shape = 128
p.scans.MF.data.num_frames = 100

# We skip saving the "prepared" data file for now. The |ptypy| data
# file is described in detail in ...
p.scans.MF.data.save = None 

# Needlees to say we need to specify a reconstruction engine. We choose
# 20 iterations of difference map
p.engines = u.Param()                                  
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 40
p.engines.engine00.numiter_contiguous = 5

# Running ptypy
# -------------

# Next we import the Ptycho class and pass the tree at level 5 which
# will make the reconstruction start immediately and will sequentially 
# initialize and use 
# all engines in ``p.engines``
from ptypy.core import Ptycho
P = Ptycho(p,level=5)



