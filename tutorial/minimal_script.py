# This tutorial explains the minimal settings to get a reconstruction
# runnig in |ptypy|. A |ptypy| script consists of two parts:

# * Creation of a parameter tree with parameters 
#   as listed in :ref:`parameters` and 
# * calling a :any:`Ptycho` instance with this parameter tree 
#   at a certain level that determines how much the Ptycho 
#   instance will do.

# Preparing the parameter tree
# ----------------------------

# We begin with opening an empty python file of arbitrary name
# in an editor of your choice, e.g.::
# 
#   $ gedit minimal_script.py

# Next we create an empty parameter tree. In |ptypy|, parameters
# are managed by the :any:`Param` class which is a subclass of Pythons
# dict type. It acts like a composite of 
# of dictionary and object, meaning that all dictionary items may be accessed as class attributes. 
from ptypy import utils as u
p = u.Param()  # root level

# We set the verbosity to a high level, in order to have information of the
# reconstruction process printed to the terminal. 
# See :py:data:`.verbose_level`.
p.verbose_level = 3                              

# We limit this reconstruction to single precision, but you may as well
# use double precision.
p.data_type = "single"                           

# We give this reconstruction the name ``'minimal'`` although it 
# will automatically choose one from the script if we put in ``None``.
# (But then the tutorial may not work on your computer as the chosen
# run name may differ from the one that this tutorial was created with)
p.run = 'minimal'

# Next, we set the home path. The :any:`Ptycho` instance will use this 
# path as base for any other file path (e.g :py:data:`.io.autosave.path` 
# or :py:data:`.io.rfile`) in the tree lacks a leading "/"
# (or "C:\\" on windows). Make sure to replace all "/" with "\\" 
# if you run the scripts on a Windows system.
p.io = u.Param()
p.io.home = "/tmp/ptypy/"

# We want an intermediate result of the reconstruction 
# to be dumped regularly every 10 reconstructions.
p.io.autosave = u.Param()
p.io.autosave.interval = 20

# In this tutorial we switch of the threaded plotting client.
p.io.autoplot = False

# Since we do not want to plot anything, we don't need the 
# interaction server either.
p.io.interaction = False

# Now we have to put in actual parameters associated with a 
# ptychographic scan.

# The ``scan`` branch of the tree holds all *common* parameters for scans
# and can be regarded as template in case of a many-scans reconstruction. 
# However, if there is only one scan, it does not matter if we specify 
# parameters for illumination or sample in this branch 
# or in ``scans``. In this tutorial
# case we do not bother to enter paramter but leave the branch empty
# (It will be filled with the defaults of :py:data:`.scan` ) .
p.scan = u.Param()

# The ``scans`` branch marks all differences for a scan with respect
# to the *default* ``scan`` branch mentioned above. 
# Different for each scan is at least the ``data`` branch
# :py:data:`.scan.data`. In this tutorial we 
# create a new scan parameter branch ``MF`` where we only specify 
# the data branch and tell |ptypy| to use scan meta data of when possible.
p.scans = u.Param()
p.scans.MF = u.Param()
p.scans.MF.if_conflict_use_meta = True
p.scans.MF.data= u.Param()

# As data source we have choosen the *'test'* source. 
# That will make |ptypy| use the internal 
# :py:class:`~ptypy.core.data.MoonFlowerScan` class.
# This class is meant for testing, and it provides/ simulates 
# diffraction patterns without using the more complex generic 
# :any:`SimScan` class.
p.scans.MF.data.source = 'test'

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
# 20 iterations of difference map algorithm.
p.engines = u.Param()                                  
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 40
p.engines.engine00.numiter_contiguous = 5

# Running ptypy
# -------------

# We import the :any:`Ptycho` class and pass the tree ``p`` at level 5.
# That will make the reconstruction start immediately after 
# and will sequentially initialize and use 
# all engines in ``p.engines``
from ptypy.core import Ptycho
P = Ptycho(p,level=5)

# From the terminal log, we note that there was an autosave every 20
# iterations and the error reduced itself from iteration to iteration.

