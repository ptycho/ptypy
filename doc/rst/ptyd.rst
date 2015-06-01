Ptypy data management
=====================


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

The PtyScan class
-----------------

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


Usage scenarios
---------------

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

Ptyd file format
----------------

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
-----------------------------

tbd

Tutorial: Creation and inspection of a .ptyd data file
------------------------------------------------------

tbd

All parameters in ``meta/`` are also in :py:data:`.scan.data`\ . When *.ptyd*-file
is loaded from within a reconstruction run, the 
 * meta [dict]:
 
   * center [array = [ 64.  64.]]
   * distance [scalar = 7.0]
   * energy [scalar = 6.2]
   * experimentID [None]
   * label [None]
   * psize [array = [ 0.000172  0.000172]]
   * shape [array = [128 128]]
   * version [string = "0.1"]
   * weight2d [128x128 float64 array]

 * common [dict]:

   * positions_scan [92x2 float64 array]
   * weight2d [128x128 float64 array]

 * info [dict]:
 
   * auto_center [None]
   * center [array = [ 64.  64.]]
   * chunk_format [string = ".chunk%02d"]
   * dfile [string = "sample.ptyd"]
   * distance [scalar = 7.0]
   * energy [scalar = 6.2]
   * experimentID [None]
   * label [None]
   * lam [None]
   * load_parallel [string = "data"]
   * min_frames [scalar = 1]
   * misfit [scalar = 0]
   * num_frames [scalar = 100]
   * orientation [None]
   * origin [string = "fftshift"]
   * positions_scan [92x2 float64 array]
   * positions_theory [None]
   * propagation [string = "farfield"]
   * psize [scalar = 0.000172]
   * rebin [scalar = 1]
   * recipe [dict]:
   * resolution [None]
   * save [string = "append"]
   * shape [array = [128 128]]
   * version [string = "0.1"]

 * chunks [dict]:

   * 0 [dict]:
   
     * data [10x128x128 int32 array]
     * indices [list = [0.000000, 1.000000, 2.000000, 3.000000,  ...]]
     * positions [10x2 float64 array]
     * weights [array = []]
     
   * 1 [dict]:
   
     * data [10x128x128 int32 array]
     * indices [list = [10.000000, 11.000000, 12.000000, 13.000000,  ...]]
     * positions [10x2 float64 array]
     * weights [array = []]
     
   * 2 [dict]:
   
     * data [10x128x128 int32 array]
     * indices [list = [20.000000, 21.000000, 22.000000, 23.000000,  ...]]
     * positions [10x2 float64 array]
     * weights [array = []]
     
   * 3 [dict]:
   
     * data [10x128x128 int32 array]
     * indices [list = [30.000000, 31.000000, 32.000000, 33.000000,  ...]]
     * positions [10x2 float64 array]
     * weights [array = []]
     * ...


References
----------

.. [h5py] http://www.h5py.org/
.. [HDF] **H**\ ierarchical **D**\ ata **F**\ ormat, `<http://www.hdfgroup.org/HDF5/>`_
