.. _ptypy_data:

*********************
Ptypy data management
*********************


For ptychography experiments we consider the following generic steps 
which the user has to complete in order to begin a ptychographic
reconstruction.

**(A)** *Conduct a scanning diffraction experiment.* 
   While or after this experiment
   is performed, the user is left with *raw images* acquired from the 
   detector, *meta data* that are in general the scanning position along
   with geometric information about the setup, e.g. photon *energy*, 
   propagation *distance*, *detector pixel size* etc.

**(B)** *Preprocess the data.*
   In this step, the user is faced to perform a subset of the actions
   comprised in the following list. The user may need to
   
   * select the appropriate region of the detector where the scattering events where counted, 
   * apply possible *frame correction* for converting the detector counts in the chosen
     diffraction frame into photon counts, e.g. flat-field and dark-field
     correction,
   * switch image orientation to match with the coordinate system of the 
     reconstruction algorithms,
   * find a suited mask to exclude invalid pixel data (hot or dead pixel, overexposure),
   * or rebin the data
   
   Finally the user needs to pack the diffraction image with the scanning positions. 

**(C)** *Save the processed data or feed the data into recontruction process.*
   In this step the user needs to save the data in a suitable format
   or provide the data directly for the reconstruction algorithm that

Data management in ptypy deals with (B) and (C) as a ptychography 
reconstruction software naturally cannot provide the actual experimental 
data. Nevertheless, the treatment of raw-data is usually very similar for 
different experiments. Consequently, ptypy provides an abstract base class
called :any:`PtyScan` which aims to help with steps (B) and (C). In order
to adapt ptypy for a specific experimental setup, the user simply
subclass :any:`PtyScan` and reimplement a subset of methods that are 
affected by the specifics of the experiemental setup 
(see `Tutorial: Subclassing PtyScan`_). 

.. _sec_ptyscan:

The PtyScan class
=================

A PtyScan instance is contstructed from a set of generic parameters,
see :py:data:`.scan.data` in the ptypy parameter tree.

:any:`PtyScan` provides the following features.

**Parallelization**
  When ptypy is run in several MPI processes, PtyScan takes care of 
  distributing the scan-point indeces among processes such that each
  process only loads the data it will later use in the reconstruction.
  Hence, the load on the network is not affected by the number of
  processes.
  The parallel behavior of :any:`PtyScan`, is controlled by the parameter 
  :py:data:`.scan.data.load_parallel`.  

**Preprocessing**
  PtyScan can handle a few of the preprocessing steps mentioned above.
  
  * Selecting a region-of-interest from the raw detector image. This
    selection is controlled by the parameters :py:data:`.scan.data.auto_center`,
    and :py:data:`.scan.data.shape` and :py:data:`.scan.data.center`.
  
  * Switching of orientation and rebinning are controlled by 
    :py:data:`.scan.data.orientation` and :py:data:`.scan.data.rebin`.
  
  * Finding a suitable mask or weight for pixel correction is left
    to the user, as this is a setup-specific implementation. 
    See :py:meth:`~ptypy.core.data.PtyScan.load_weight`,
    :py:meth:`~ptypy.core.data.PtyScan.load_common`,
    :py:meth:`~ptypy.core.data.PtyScan.load`
    and :py:meth:`~ptypy.core.data.PtyScan.correct`
    for detailed explanations.
    
**Packaging**
  PtyScan packs the processed *data* together with the used scan point 
  *indices*, scan *positions* and a *weight* (=mask) and geometric *meta*
  information. This package is requested by the managing instance 
  :py:class:`~ptypy.core.manager.ModelManager` with the call 
  :py:meth:`~ptypy.core.manager.ModelManager.new_data`.
  
  The minimum number of data frames passed to each process on a *new_data()*
  call is set by :py:data:`~.scan.data.min_frames`. The total number
  of frames processed for a scan is set by :py:data:`~.scan.data.num_frames`.
  
  If positions or other meta data is not extracted from other files, 
  the user may set the photon energy with :py:data:`.scan.data.energy`,
  the propagation distance from sample to detector with 
  :py:data:`.scan.data.distance` and the detector pixelsize with
  :py:data:`.scan.data.psize`.

**Storage**
  PtyScan and its subclass are capable of storing the data in an 
  *hfd5*-compatible [HDF]_ file format. The data file names have a custom 
  suffix: *.ptyd*.
  
  A detailed overview of the *.ptyd* data file tree is written below in 
  the section `Ptyd file format`_
  
  The saving behavior of :any:`PtyScan`, is controlled by the parameters 
  :py:data:`.scan.data.save` and :py:data:`.scan.data.chunk_format`
  
  .. note::
     Although *h5py* [h5py]_ supports parallel write, this feature is not 
     used in ptypy. At the moment, the mpi nodes send the ir 

.. _ptyd_scenarios:

Usage scenarios
===============

Ptypy provides support for three data usage cases.

**Beamline integretion use.** 
  
  In this use case 
  
  sadfsg
  
  .. figure:: ../img/data_case_integrated.png
     :width: 70 %
     :figclass: highlights
     :name: case_integrated

     This is a test of a figure plot
     
**Post preparation use.**
  bla bla

  .. figure:: ../img/data_case_prepared.png
     :width: 70 %
     :figclass: highlights
     :name: case_prepared
     
     This is a test of a figure plot
     
**Simultaneous acquisition and loading.**
  bla bla
  
  .. figure:: ../img/data_case_flyscan.png
     :width: 70 %
     :figclass: highlights
     :name: case_flyscan

     This is a test of a figure plot

Ptypy uses the python module **h5py** [h5py]_ to store and load data in the
**H**\ ierarchical **D**\ ata **F**\ ormat [HDF]_ . The HDF resembles very 
much a directory/file tree of todays operating systems, while the "files"
are (multidimensonial) datasets. 

.. _ptyd_file:

Ptyd file format
================

Ptypy stores and loads the (processed) experimental in a file with ending
*.ptyd*, which is a hdf5-file with a data tree of very simple nature. 
Comparable to tagged image file formats like *.edf* or *.tiff*, the ``ptyd`` data file seperates
meta information (stored in ``meta/``) from the actual data payload 
(stored in ``chunks/``). A schematic overview of the data tree is depicted below.

::
   
   *.ptyd/
     
         meta/
            
            [general parameters; optional but very useful]
            version     : str
            num_frames  : int
            label       : str
            
            [geometric porameters; all optional]
            shape       : int or (int,int)
            energy      : float, optional
            distance    : float, optional
            center      : (float,float) or None, optional
            psize       : float or (float,float), optional
            propagation : "farfield" or "nearfield", optional
            ...
            
         chunks/
         
            0/
              data      : array(M,N,N) of float
              indices   : array(M) of int, optional
              positions : array(M ,2) of float
              weights   : same shape as data or empty
            1/
              ...
            2/
              ...
            ...

All parameters of ``meta/`` are a subset of :py:data:`.scan.data`\ .
Omitting any of these parameters or setting the value of the dataset to 
``'None'`` has the same effect.

The first set of parameters

::
   
   version     : str 
   num_frames  : int 
   label       : str 

are general (optional) parameters.
 
  * ``version`` is ptypy version this dataset was prepared with
    (current version is |version|, see :py:data:`~.scan.data.version`).
  * ``label`` is a custom user label. Choose a unique label to your liking.
  * ``num_frames`` indicates how many diffraction image frames are 
    expected in the dataset (see :py:data:`~.scan.data.num_frames`)
    It is important to set this parameter when the data acquisition is not
    finished but the reconstruction has already started. If the dataset
    is complete, the loading class :any:`PtydScan` retrieves the 
    total number of frames from the payload ``chunks/``
    
The next set of optional parameters are

::

   shape       : int or (int,int)
   energy      : float
   distance    : float
   center      : (float,float)
   psize       : float or (float,float)
   propagation : "farfield" or "nearfield"

which refer to the experimental scanning geometry. 

  * ``shape`` 
    (see :py:data:`.scan.data.shape`)
  * ``energy`` 
    (see :py:data:`.scan.data.energy` or :py:data:`.scan.geometry.energy`)
  * ``distance`` 
    (see :py:data:`.scan.data.distance`)
  * ``center``      : (float,float)
    (see :py:data:`.scan.data.center`)
  * ``psize``       : float or (float,float)
    (see :py:data:`.scan.data.psize`)
  * ``propagation`` : "farfield" or "nearfield"
    (see :py:data:`.scan.data.propagation`)

Finally these parameters will be digested by the 
:py:mod:`~ptypy.core.geometry` module in order to provide a suited propagator.

.. note::
   
   As you may have already noted, there are three ways to specify the 
   geometry of the experiment. 
   
   ::
   
      bla



As walking the data tree and extracting the data from the *hdf5* file 
is a bit cumbersome with h5py, there are a few convenience function in the 
:py:mod:`ptypy.io.h5rw` module.

Tutorial: Subclassing PtyScan
=============================

.. note::
   This tutorial was generated from the python source :file:`ptypy/tutorial/subclassptyscan.py` using :file:`ptypy/doc/script2rst.py`.

.. _subclassptyscan:

Tutorial : Subclassing PtyScan
==============================

In this tutorial, we learn how to subclass :any:`PtyScan` to make 
ptypy work with any experimental setup.

This tutorial can be used as a direct follow-up to :ref:`simupod` 
if section :ref:`store` was completed

Again, the imports first.

::

   >>> import matplotlib as mpl
   >>> import numpy as np
   >>> import ptypy
   >>> from ptypy import utils as u
   >>> plt = mpl.pyplot
   >>> import sys

For this tutorial we assume, that the data and meta information is 
in this path:

::

   >>> save_path = '/tmp/ptypy/sim/'

Furthermore, we assume that a file about the experimental geometry is
located at 

::

   >>> geofilepath = save_path+ 'geometry.txt'
   >>> print geofilepath
   /tmp/ptypy/sim/geometry.txt
   
and has contents of the following form

::

   >>> print ''.join([line for line in open(geofilepath,'r')])
   distance 1.5000e-01
   energy 2.3319e-03
   psize 2.4000e-05
   shape 256
   
   

The scanning positions are in 

::

   >>> positionpath = save_path+ 'positions.txt'
   >>> print positionpath
   /tmp/ptypy/sim/positions.txt
   

with a list of positions for vertical and horizontanl movement and the
image frame from the "camera" 

::

   >>> print ''.join([line for line in open(positionpath,'r')][:6])+'....'
   ccd/diffraction_0000.npy 1.4658e-03 2.0175e-03
   ccd/diffraction_0001.npy 1.8532e-03 1.6686e-03
   ccd/diffraction_0002.npy -1.7546e-03 1.1135e-03
   ccd/diffraction_0003.npy -1.4226e-03 1.5149e-03
   ccd/diffraction_0004.npy -2.0740e-03 1.3049e-04
   ccd/diffraction_0005.npy -1.9764e-03 6.4218e-04
   ....
   

Writing a subclass
------------------

A subclass of :any:`PtyScan` takes the same input parameter 
tree as PtyScan itself, i.e :py:data:`.scan.data`. As the subclass
will most certainly require additional parameters, there has to be 
a flexible additional container. For PtyScan, that is the 
:py:data:`.scan.data.recipe` parameter. A subclass must extract all 
additional parameters from this source and, in script, you fill
the recipe with the appropriate items.

In this case we can assume that the only parameter of the recipe
is the base path ``/tmp/ptypy/sim/``\ . Hence we write

::

   >>> RECIPE = u.Param()
   >>> RECIPE.base_path = '/tmp/ptypy/sim/'

Now we import the deafult generic parameter set from

::

   >>> from ptypy.core.data import PtyScan
   >>> DEFAULT = PtyScan.DEFAULT.copy()

This would be the perfect point to change any default value.
For sure we set.

::

   >>> DEFAULT.recipe = RECIPE

A default data file location may be handy too and we allow saving of
data in a single file. And since we now it is simulated dat we do not
have to find the optical axes in the diffraction pattern with
the help of auto_center

::

   >>> DEFAULT.dfile = '/tmp/ptypy/sim/npy.ptyd'
   >>> DEFAULT.auto_center = False

Our defaults are now

::

   >>> print u.verbose.report(DEFAULT,noheader=True)
   * id3VE7SOA57G           : ptypy.utils.parameters.Param(19)
     * positions_theory     : None
     * auto_center          : False
     * chunk_format         : .chunk%02d
     * min_frames           : 1
     * orientation          : None
     * num_frames           : None
     * energy               : None
     * center               : None
     * recipe               : ptypy.utils.parameters.Param(1)
       * base_path          : /tmp/ptypy/sim/
     * psize                : None
     * label                : None
     * load_parallel        : data
     * shape                : None
     * rebin                : None
     * experimentID         : None
     * version              : 0.1
     * save                 : None
     * dfile                : /tmp/ptypy/sim/npy.ptyd
     * distance             : None
   
   

The simplest subclass of PtyScan would look like this

::

   >>> class NumpyScan(PtyScan):
   >>>     # We overwrite the DEFAULT with the new DEFAULT.
   >>>     DEFAULT = DEFAULT
   >>>     
   >>>     def __init__(self,pars=None, **kwargs):
   >>>         # In init we need to call the parent.
   >>>         super(NumpyScan, self).__init__(pars, **kwargs)

Of course this class does nothing special beyond PtyScan.

An additional step of initialisation would be to retrieve 
the geometric information that we stored in ``geofilepath`` and update
the input parameters with it.

We write a tiny file parser.

::

   >>> def extract_geo(base_path):
   >>>     out = {}
   >>>     with open(base_path+'geometry.txt') as f:
   >>>         for line in f:
   >>>             key, value = line.strip().split()
   >>>             out[key]=eval(value)
   >>>     return out

We test it.

::

   >>> print extract_geo(save_path)
   {'distance': 0.15, 'energy': 0.0023319, 'shape': 256, 'psize': 2.4e-05}
   

That seems to work. We can integrate this parser into 
the initialisation as we assume that this small access can be 
done by all MPI nodes without data access problems. Hence,
our subclass becomes

::

   >>> class NumpyScan(PtyScan):
   >>>     # We overwrite the DEFAULT with the new DEFAULT.
   >>>     DEFAULT = DEFAULT
   >>>     
   >>>     def __init__(self,pars=None, **kwargs):
   >>>         p = DEFAULT.copy(depth=2)
   >>>         p.update(pars) 
   >>>         
   >>>         with open(p.recipe.base_path+'geometry.txt') as f:
   >>>             for line in f:
   >>>                 key, value = line.strip().split()
   >>>                 # we only replace Nones or missing keys
   >>>                 if p.get(key) is None:
   >>>                     p[key]=eval(value)
   >>>         
   >>>         super(NumpyScan, self).__init__(p, **kwargs)

Good! Next, we need to implement how the class finds out about
the positions in the scan. The method 
:py:meth:`~ptypy.core.data.PtyScan.load_positions` can be used
for this purpose.

::

   >>> print PtyScan.load_positions.__doc__
   
           **Override in subclass for custom implementation**
           
           *Called in* :py:meth:`initialize`
           
           Loads all positions for all diffraction patterns in this scan. 
           The positions loaded here will be available by all processes 
           through the attribute ``self.positions``. If you specify position
           on a per frame basis in :py:meth:`load` , this function has no 
           effect.
           
           If theoretical positions :py:data:`positions_theory` are 
           provided in the initial parameter set :py:data:`DEFAULT`, 
           specifyiing positions here has NO effect and will be ignored.
           
           The purpose of this function is to avoid reloading and parallel
           reads on files that may require intense parsing to retrieve the
           information, e.g. long SPEC log files. If parallel reads or 
           log file parsing for each set of frames is not a time critical
           issue of the subclass, reimplementing this function can be ignored
           and it is recommended to only reimplement the :py:meth:`load` 
           method.
           
           If `load_parallel` is set to `all` or common`, this function is 
           executed by all nodes, otherwise the master node executes this
           function and braodcasts the results to other nodes. 
           
           Returns
           -------
           positions : ndarray
               A (N,2)-array where *N* is the number of positions.
               
           Note
           ----
           Be aware that this method sets attribute :py:attr:`num_frames`
           in the following manner.
           
           * If ``num_frames == None`` : ``num_frames = N``.
           * If ``num_frames < N`` , no effect.
           * If ``num_frames > N`` : ``num_frames = N``.
            
           
   

The parser for the positions file would look like this.

::

   >>> def extract_pos(base_path):
   >>>     pos = []
   >>>     files =[]
   >>>     with open(base_path+'positions.txt') as f:
   >>>         for line in f:
   >>>             fname, y, x = line.strip().split()
   >>>             pos.append((eval(y),eval(x)))
   >>>             files.append(fname)
   >>>     return files,pos

And the test:

::

   >>> files, pos = extract_pos(save_path)
   >>> print files[:2]
   ['ccd/diffraction_0000.npy', 'ccd/diffraction_0001.npy']
   
   >>> print pos[:2]
   [(0.0014658, 0.0020175), (0.0018532, 0.0016686)]
   


::

   >>> class NumpyScan(PtyScan):
   >>>     # We overwrite the DEFAULT with the new DEFAULT.
   >>>     DEFAULT = DEFAULT
   >>>     
   >>>     def __init__(self,pars=None, **kwargs):
   >>>         p = DEFAULT.copy(depth=2)
   >>>         p.update(pars) 
   >>>         
   >>>         with open(p.recipe.base_path+'geometry.txt') as f:
   >>>             for line in f:
   >>>                 key, value = line.strip().split()
   >>>                 # we only replace Nones or missing keys
   >>>                 if p.get(key) is None:
   >>>                     p[key]=eval(value)
   >>>         
   >>>         super(NumpyScan, self).__init__(p, **kwargs)
   >>>         # all input data is now in self.info
   >>>         
   >>>     def load_positions(self):
   >>>         # the base path is now stored in 
   >>>         base_path = self.info.recipe.base_path
   >>>         with open(base_path+'positions.txt') as f:
   >>>             for line in f:
   >>>                 fname, y, x = line.strip().split()
   >>>                 pos.append((eval(y),eval(x)))
   >>>                 files.append(fname)
   >>>         return np.asarray(pos)

One nice thing about rewriting ``self.load_positions`` is that the 
the maximum number of frames will be set and we do not need to
manually adapt :py:meth:`~ptypy.core.data.PtyScan.check`

The last step is to overwrite the actual loading of data.
Loading happens (MPI-compatible) in 
:py:meth:`~ptypy.core.data.PtyScan.load`

::

   >>> print PtyScan.load.__doc__
   
           **Override in subclass for custom implementation**
           
           Loads data according to node specific scanpoint indeces that have 
           been determined by :py:class:`LoadManager` or otherwise
           
           Returns
           -------
           raw, positions, weight : dict
               Dictionaries whose keys are the given scan point `indices` 
               and whose values are the respective frame / position according 
               to the scan point index. `weight` and `positions` may be empty
               
           Note
           ----
           This is the *most* important method to change when subclassing
           :any:`PtyScan`. Most often it suffices to override the constructor
           and this method to createa subclass of suited for a specific 
           experiment.
           
   

Load seems a bit more complex than ``self.load_positions`` for its 
return values. However, we can opt-out of providing weights (masks)
and positions, as we have already adapted ``self.load_positions``
and we there were no bad pixels in the (linear) detector

The final subclass looks like this.

::

   >>> class NumpyScan(PtyScan):
   >>>     # We overwrite the DEFAULT with the new DEFAULT.
   >>>     DEFAULT = DEFAULT
   >>>     
   >>>     def __init__(self,pars=None, **kwargs):
   >>>         p = DEFAULT.copy(depth=2)
   >>>         p.update(pars) 
   >>>         
   >>>         with open(p.recipe.base_path+'geometry.txt') as f:
   >>>             for line in f:
   >>>                 key, value = line.strip().split()
   >>>                 # we only replace Nones or missing keys
   >>>                 if p.get(key) is None:
   >>>                     p[key]=eval(value)
   >>>         
   >>>         super(NumpyScan, self).__init__(p, **kwargs)
   >>>         # all input data is now in self.info
   >>>         
   >>>     def load_positions(self):
   >>>         # the base path is now stored in
   >>>         pos=[] 
   >>>         base_path = self.info.recipe.base_path
   >>>         with open(base_path+'positions.txt') as f:
   >>>             for line in f:
   >>>                 fname, y, x = line.strip().split()
   >>>                 pos.append((eval(y),eval(x)))
   >>>                 files.append(fname)
   >>>         return np.asarray(pos)
   >>>     
   >>>     def load(self,indices):
   >>>         raw = {}
   >>>         bp = self.info.recipe.base_path
   >>>         for ii in indices:
   >>>             raw[ii] = np.load(bp+'ccd/diffraction_%04d.npy' % ii )
   >>>         return raw, {},{}

Loading the data
----------------

With the subclass we create a scan only using defaults

::

   >>> NPS = NumpyScan()
   >>> NPS.initialize()

In order to process the data. We need to call 
:py:meth:`~ptypy.core.data.PtyScan.auto` with the chunk size
as arguments. It returns a data chunk that we can inspect
with :py:func:`ptypy.utils.verbose.report`. The information is 
concetanated, but the length of iterables or dicts is always indicated
in parantheses.

::

   >>> print u.verbose.report(NPS.auto(80),noheader=True)
   * id3VE7SNKS2G           : dict(2)
     * common               : ptypy.utils.parameters.Param(9)
       * distance           : 0.15
       * center             : [array = [ 128.  128.]]
       * energy             : 0.0023319
       * psize              : [array = [  2.40000000e-05   2.40000000e-05]]
       * label              : None
       * shape              : [array = [256 256]]
       * version            : 0.1
       * experimentID       : None
       * weight2d           : [256x256 bool array]
     * iterable             : list(80)
       * id3VE7SNLQQ0       : dict(4)
         * index            : 0
         * data             : [256x256 int32 array]
         * mask             : [256x256 bool array]
         * position         : [array = [ 0.0014658  0.0020175]]
       * id3VE7SNN1VO       : dict(4)
         * index            : 1
         * data             : [256x256 int32 array]
         * mask             : [256x256 bool array]
         * position         : [array = [ 0.0018532  0.0016686]]
       * id3VE7SNN2H8       : dict(4)
         * index            : 2
         * data             : [256x256 int32 array]
         * mask             : [256x256 bool array]
         * position         : [array = [-0.0017546  0.0011135]]
       * id3VE7SNLRK8       : dict(4)
         * index            : 3
         * data             : [256x256 int32 array]
         * mask             : [256x256 bool array]
         * position         : [array = [-0.0014226  0.0015149]]
       * id3VE7SNLEQ0       : dict(4)
         * index            : 4
         * data             : [256x256 int32 array]
         * mask             : [256x256 bool array]
         * position         : [array = [-0.002074    0.00013049]]
       * ...                : ....
   
   
   >>> print u.verbose.report(NPS.auto(80),noheader=True)
   * id3VE7SNN15G           : dict(2)
     * common               : ptypy.utils.parameters.Param(9)
       * distance           : 0.15
       * center             : [array = [ 128.  128.]]
       * energy             : 0.0023319
       * psize              : [array = [  2.40000000e-05   2.40000000e-05]]
       * label              : None
       * shape              : [array = [256 256]]
       * version            : 0.1
       * experimentID       : None
       * weight2d           : [256x256 bool array]
     * iterable             : list(34)
       * id3VE7SNLQH8       : dict(4)
         * index            : 80
         * data             : [256x256 int32 array]
         * mask             : [256x256 bool array]
         * position         : [array = [-0.0021597 -0.0012469]]
       * id3VE7SNJBK8       : dict(4)
         * index            : 81
         * data             : [256x256 int32 array]
         * mask             : [256x256 bool array]
         * position         : [array = [-0.0023717  -0.00077061]]
       * id3VE7SNMEQ0       : dict(4)
         * index            : 82
         * data             : [256x256 int32 array]
         * mask             : [256x256 bool array]
         * position         : [array = [-0.0024801  -0.00026067]]
       * id3VE7SNJ82G       : dict(4)
         * index            : 83
         * data             : [256x256 int32 array]
         * mask             : [256x256 bool array]
         * position         : [array = [-0.0024801   0.00026067]]
       * id3VE7SNJA8G       : dict(4)
         * index            : 84
         * data             : [256x256 int32 array]
         * mask             : [256x256 bool array]
         * position         : [array = [-0.00051848 -0.0024393 ]]
       * ...                : ....
   
   

We observe the the second chunk was not 80 frames deep but 34
as we only had 114 frames of data.

So where is the *.ptyd* data-file? As default, PtyScan does not
actually save data. We have to manually activate it in in the 
input paramaters.

::

   >>> data = NPS.DEFAULT.copy(depth=2)
   >>> data.save = 'append'
   >>> NPS = NumpyScan(pars = data)
   >>> NPS.initialize()
   WARNING root - File /tmp/ptypy/sim/npy.ptyd already exist. Renamed to /tmp/ptypy/sim/npy.ptyd.old
   

::

   >>> for i in range(50):
   >>>     msg = NPS.auto(20)
   >>>     if msg==NPS.EOS:
   >>>         break

We can analyse the saved ``npy.ptyd`` with 
:py:func:`~ptypy.io.h5IO.h5info`

::

   >>> from ptypy.io import h5info
   >>> print h5info(NPS.info.dfile)
   File created : Tue Jun  9 23:20:19 2015
    * chunks [dict 6]:
        * 0 [dict 4]:
            * data [20x256x256 int32 array]
            * indices [list = [0.000000, 1.000000, 2.000000, 3.000000,  ...]]
            * positions [20x2 float64 array]
            * weights [array = []]
        * 1 [dict 4]:
            * data [20x256x256 int32 array]
            * indices [list = [20.000000, 21.000000, 22.000000, 23.000000,  ...]]
            * positions [20x2 float64 array]
            * weights [array = []]
        * 2 [dict 4]:
            * data [20x256x256 int32 array]
            * indices [list = [40.000000, 41.000000, 42.000000, 43.000000,  ...]]
            * positions [20x2 float64 array]
            * weights [array = []]
        * 3 [dict 4]:
            * data [20x256x256 int32 array]
            * indices [list = [60.000000, 61.000000, 62.000000, 63.000000,  ...]]
            * positions [20x2 float64 array]
            * weights [array = []]
        * 4 [dict 4]:
            * data [20x256x256 int32 array]
            * indices [list = [80.000000, 81.000000, 82.000000, 83.000000,  ...]]
            * positions [20x2 float64 array]
            * weights [array = []]
        * 5 [dict 4]:
            * data [14x256x256 int32 array]
            * indices [list = [100.000000, 101.000000, 102.000000, 103.000000,  ...]]
            * positions [14x2 float64 array]
            * weights [array = []]
    * info [dict 20]:
        * auto_center [scalar = False]
        * center [array = [128 128]]
        * chunk_format [string = ".chunk%02d"]
        * dfile [string = "/tmp/ptypy/sim/npy.ptyd"]
        * distance [scalar = 0.15]
        * energy [scalar = 0.0023319]
        * experimentID [None]
        * label [None]
        * load_parallel [string = "data"]
        * min_frames [scalar = 1]
        * num_frames [None]
        * orientation [None]
        * positions_scan [114x2 float64 array]
        * positions_theory [None]
        * psize [scalar = 2.4e-05]
        * rebin [scalar = 1]
        * recipe [Param 1]:
            * base_path [string = "/tmp/ptypy/sim/"]
        * save [string = "append"]
        * shape [array = [256 256]]
        * version [string = "0.1"]
    * meta [dict 9]:
        * center [array = [ 128.  128.]]
        * distance [scalar = 0.15]
        * energy [scalar = 0.0023319]
        * experimentID [None]
        * label [None]
        * psize [array = [  2.40000000e-05   2.40000000e-05]]
        * shape [array = [256 256]]
        * version [string = "0.1"]
        * weight2d [256x256 bool array]
   
   None
   





.. [h5py] http://www.h5py.org/
.. [HDF] **H**\ ierarchical **D**\ ata **F**\ ormat, `<http://www.hdfgroup.org/HDF5/>`_
