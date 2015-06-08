.. note::
   This tutorial was generated from the python source :file:`ptypy/tutorial/minimal_script.py` using :file:`ptypy/doc/script2rst.py`.

This tutorial explains the minimal settings to get a reconstruction
runnig in |ptypy|. A |ptypy| script consists of 2 parts, creation of
a parameter tree with parameters as listed in :ref:`parameters` and 
finally calling a :any:`Ptycho` instance with these parameter tree 
and a certain level that determines how much the Ptycho 
instance will do.

Preparing the parameter tree
----------------------------

We begin with creating an empty parameter tree. In |ptypy| parameters
are managed by the :any:`Param` class which is a subclass of Pythons
dict type and acts like a composite of 
of dictionary and object: All items can be accessed as class attributes. 

::

   >>> from ptypy import utils as u
   >>> p = u.Param()  # root level

We set the verbosity to a high level, such that we information on the
reconstruction process are printed into the terminal. 
See :py:data:`.verbose_level`.

::

   >>> p.verbose_level = 3

We limit this reconstruction to single precision, but yoy may as well
use double precision

::

   >>> p.data_type = "single"

We give this reconstruction the name ``'minimal'`` although it 
will automatically choose one from the script if we put in ``None``.

::

   >>> p.run = 'minimal'

We set the home path. The reconstruction will save / dump anything
into this base path if any other path in the tree lacks a leading "/" 
(or "C:\" on windows). Make sure to replace all "/" with "\" if you run
the scripts on a Windows system.

::

   >>> p.io = u.Param()
   >>> p.io.home = "/tmp/ptypy/"

We want a state of the reconstruction to be dumped regularly every
10 reconstructions

::

   >>> p.io.autosave = u.Param()
   >>> p.io.autosave.interval = 10

In this tutorial we switch of the threaded plotting

::

   >>> p.io.autoplot = False

Since we are not plotting we do not need the interaction server either.

::

   >>> p.io.interaction = False

This branch of the tree would hold all *common* parameters for scans. 
If there is only one scan, it does not matter if we specify parameters
for illumination or sample in this branch or in ``scans``. In this
case we do not bother to enter paramter but leave the branch empty
(That will fill it with the defaults :py:data:`.scan` ) 

::

   >>> p.scan = u.Param()

The ``scans`` branch encloses all differences for a scan with respect
to the ``scan`` branch mentioned above. For sure different for each 
scan is the ``data`` branch :py:data:`.scan.data`. In this case we 
create a new scan paramter branch ``MF``. We only specify the data
branch and tell |ptypy| to use scan meta data when possible.

::

   >>> p.scans = u.Param()
   >>> p.scans.MF = u.Param()
   >>> p.scans.MF.if_conflict_use_meta = True
   >>> p.scans.MF.data= u.Param()

As data source we choose the *'test'* source. This will make ptypy
use the internal :py:class:`~ptypy.core.data.MoonFlowerScan` class.
This class is a test class, that provides/ simulates diffraction 
patterns without explicitly using the generic :any:`SimScan` class.

::

   >>> p.scans.MF.data.source = 'test'

We set the diffraction frame shape to a small value and only allow 
a hundred diffraction patterns at max. The 
:py:class:`~ptypy.core.data.MoonFlowerScan` instance will balance the
diffraction patterns accordingly.

::

   >>> p.scans.MF.data.shape = 128
   >>> p.scans.MF.data.num_frames = 100

We skip saving the "prepared" data file for now. The |ptypy| data
file is described in detail in ...

::

   >>> p.scans.MF.data.save = None

Needlees to say we need to specify a reconstruction engine. We choose
20 iterations of difference map

::

   >>> p.engines = u.Param()
   >>> p.engines.engine00 = u.Param()
   >>> p.engines.engine00.name = 'DM'
   >>> p.engines.engine00.numiter = 20

Running ptypy
-------------

Next we import the Ptycho class and pass the tree at level 5 which
will make the reconstruction start immediately and will sequentially 
initialize and use 
all engines in ``p.engines``

::

   >>> from ptypy.core import Ptycho
   >>> P = Ptycho(p,level=5)
   Verbosity set to 3
   Data type:               single
   
   ---- Ptycho init level 1 -------------------------------------------------------
   Model: sharing probe between scans (one new probe every 1 scan)
   Model: sharing probe between scans (one new probe every 1 scan)
   
   ---- Ptycho init level 2 -------------------------------------------------------
   Prepared 92 positions
   Processing new data.
   ---- Enter PtyScan.initialixe() ------------------------------------------------
                Common weight : True
                       shape = (128, 128)
   All experimental positions : True
                       shape = (92, 2)
   Scanning positions (92) are fewer than the desired number of scan points (100).
   Resetting `num_frames` to lower value
   ---- Leaving PtyScan.initialixe() ----------------------------------------------
   ROI center is [ 64.  64.], automatic guess is [ 63.43478261  63.55434783].
   Feeding data chunk
   Importing data from MF as scan MF.
   End of scan reached
   End of scan reached
   
   --- Scan MF photon report ---
   Total photons   : 3.66e+09 
   Average photons : 3.98e+07
   Maximum photons : 7.13e+07
   -----------------------------
   
   ---- Creating PODS -------------------------------------------------------------
   Found these probes : 
   Found these objects: 
   Process 0 created 92 new PODs, 1 new probes and 1 new objects.
   
   ---- Probe initialization ------------------------------------------------------
   Initializing probe storage S00G00 using scan MF
   Found no photon count for probe in parameters.
   Using photon count 7.13e+07 from photon report
   
   ---- Object initialization -----------------------------------------------------
   Initializing object storage S00G00 using scan MF
   Simulation resource is object transmission
   
   ---- Creating exit waves -------------------------------------------------------
   
   Process #0 ---- Total Pods 92 (92 active) ----
   --------------------------------------------------------------------------------
   (C)ontnr : Memory : Shape            : Pixel size      : Dimensions      : Views
   (S)torgs : (MB)   : (Pixel)          : (meters)        : (meters)        : act. 
   --------------------------------------------------------------------------------
   Cprobe   :    0.1 : complex64
   S00G00   :    0.1 :        1*128*128 :   6.36*6.36e-08 :   8.14*8.14e-06 :    92
   Cmask    :    1.5 :   bool
   S0000    :    1.5 :       92*128*128 :   1.72*1.72e-04 :   2.20*2.20e-02 :    92
   Cexit    :   12.1 : complex64
   S0000G00 :   12.1 :       92*128*128 :   6.36*6.36e-08 :   8.14*8.14e-06 :    92
   Cobj     :    1.3 : complex64
   S00G00   :    1.3 :        1*394*408 :   6.36*6.36e-08 :   2.51*2.60e-05 :    92
   Cdiff    :    6.0 : float32
   S0000    :    6.0 :       92*128*128 :   1.72*1.72e-04 :   2.20*2.20e-02 :    92
   
   
   
   ---- Ptycho init level 3 -------------------------------------------------------
   
   ---- Ptycho init level 4 -------------------------------------------------------
   
   ==== Starting DM-algoritm. =====================================================
   
   Parameter set:
   * id3VVC8ATELO           : ptypy.utils.parameters.Param(16)
     * clip_object          : None
     * fourier_relax_factor : 0.05
     * numiter_contiguous   : 1
     * overlap_converge_... : 0.1
     * probe_update_start   : 0
     * probe_inertia        : 0.001
     * name                 : DM
     * subpix_start         : 0
     * update_object_first  : True
     * obj_smooth_std       : None
     * alpha                : 1
     * overlap_max_itera... : 10
     * object_inertia       : 0.1
     * numiter              : 20
     * probe_support        : 0.8
     * subpix               : linear
   ================================================================================
   ---------------------------------- Autosaving ----------------------------------
   Generating copies of probe, object and parameters and runtime
   Saving to /tmp/ptypy/dumps/minimal/minimal_None_0000.ptyr
   --------------------------------------------------------------------------------
   Time spent in Fourier update: 0.23
   Iteration (Overlap) #00:  change in probe is 1.000
   Iteration (Overlap) #01:  change in probe is 0.048
   Iteration #1 of DM :: Time 0.31
   Errors :: Fourier 1.12e+03, Photons 1.15e+05, Exit 1.02e+03
   Time spent in Fourier update: 0.23
   Iteration (Overlap) #00:  change in probe is 0.080
   Iteration #2 of DM :: Time 0.27
   Errors :: Fourier 1.04e+03, Photons 1.67e+04, Exit 5.40e+02
   Time spent in Fourier update: 0.24
   Iteration (Overlap) #00:  change in probe is 0.114
   Iteration (Overlap) #01:  change in probe is 0.065
   Iteration #3 of DM :: Time 0.32
   Errors :: Fourier 1.35e+03, Photons 7.96e+03, Exit 8.17e+02
   Time spent in Fourier update: 0.23
   Iteration (Overlap) #00:  change in probe is 0.110
   Iteration (Overlap) #01:  change in probe is 0.068
   Iteration #4 of DM :: Time 0.32
   Errors :: Fourier 1.33e+03, Photons 2.80e+03, Exit 8.59e+02
   Time spent in Fourier update: 0.24
   Iteration (Overlap) #00:  change in probe is 0.096
   Iteration #5 of DM :: Time 0.28
   Errors :: Fourier 1.21e+03, Photons 2.28e+03, Exit 8.41e+02
   Time spent in Fourier update: 0.23
   Iteration (Overlap) #00:  change in probe is 0.082
   Iteration #6 of DM :: Time 0.27
   Errors :: Fourier 1.13e+03, Photons 1.40e+03, Exit 7.98e+02
   Time spent in Fourier update: 0.24
   Iteration (Overlap) #00:  change in probe is 0.069
   Iteration #7 of DM :: Time 0.28
   Errors :: Fourier 1.06e+03, Photons 1.42e+03, Exit 7.36e+02
   Time spent in Fourier update: 0.23
   Iteration (Overlap) #00:  change in probe is 0.060
   Iteration #8 of DM :: Time 0.27
   Errors :: Fourier 1.01e+03, Photons 9.68e+02, Exit 6.87e+02
   Time spent in Fourier update: 0.24
   Iteration (Overlap) #00:  change in probe is 0.049
   Iteration #9 of DM :: Time 0.28
   Errors :: Fourier 9.78e+02, Photons 9.08e+02, Exit 6.62e+02
   Time spent in Fourier update: 0.23
   Iteration (Overlap) #00:  change in probe is 0.048
   Iteration #10 of DM :: Time 0.27
   Errors :: Fourier 9.48e+02, Photons 8.36e+02, Exit 5.96e+02
   ---------------------------------- Autosaving ----------------------------------
   Generating copies of probe, object and parameters and runtime
   Saving to /tmp/ptypy/dumps/minimal/minimal_DM_0010.ptyr
   --------------------------------------------------------------------------------
   Time spent in Fourier update: 0.24
   Iteration (Overlap) #00:  change in probe is 0.046
   Iteration #11 of DM :: Time 0.28
   Errors :: Fourier 9.28e+02, Photons 7.88e+02, Exit 5.47e+02
   Time spent in Fourier update: 0.23
   Iteration (Overlap) #00:  change in probe is 0.046
   Iteration #12 of DM :: Time 0.27
   Errors :: Fourier 9.27e+02, Photons 7.41e+02, Exit 5.14e+02
   Time spent in Fourier update: 0.24
   Iteration (Overlap) #00:  change in probe is 0.047
   Iteration #13 of DM :: Time 0.28
   Errors :: Fourier 9.16e+02, Photons 6.01e+02, Exit 4.85e+02
   Time spent in Fourier update: 0.23
   Iteration (Overlap) #00:  change in probe is 0.047
   Iteration #14 of DM :: Time 0.27
   Errors :: Fourier 9.02e+02, Photons 5.41e+02, Exit 4.49e+02
   Time spent in Fourier update: 0.24
   Iteration (Overlap) #00:  change in probe is 0.049
   Iteration #15 of DM :: Time 0.28
   Errors :: Fourier 8.81e+02, Photons 5.23e+02, Exit 4.12e+02
   Time spent in Fourier update: 0.23
   Iteration (Overlap) #00:  change in probe is 0.048
   Iteration #16 of DM :: Time 0.28
   Errors :: Fourier 8.58e+02, Photons 4.76e+02, Exit 3.77e+02
   Time spent in Fourier update: 0.23
   Iteration (Overlap) #00:  change in probe is 0.047
   Iteration #17 of DM :: Time 0.28
   Errors :: Fourier 8.22e+02, Photons 4.56e+02, Exit 3.44e+02
   Time spent in Fourier update: 0.23
   Iteration (Overlap) #00:  change in probe is 0.042
   Iteration #18 of DM :: Time 0.27
   Errors :: Fourier 7.86e+02, Photons 3.97e+02, Exit 3.00e+02
   Time spent in Fourier update: 0.23
   Iteration (Overlap) #00:  change in probe is 0.041
   Iteration #19 of DM :: Time 0.28
   Errors :: Fourier 7.50e+02, Photons 3.49e+02, Exit 2.66e+02
   Time spent in Fourier update: 0.23
   Iteration (Overlap) #00:  change in probe is 0.039
   Iteration #20 of DM :: Time 0.27
   Errors :: Fourier 7.06e+02, Photons 2.86e+02, Exit 2.30e+02
   Generating shallow copies of probe, object and parameters and runtime
   Saving to /tmp/ptypy/recons/minimal/minimal_DM.ptyr
   



