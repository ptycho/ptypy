Ptypy data structure
====================

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
            shape       : int or (int,int)
            label       : str
            
            [geometric porameters; all optional] 
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

All parameters of ``meta/`` are also in :py:data:`.scan.data`\ .
The first set of parameters

::
   
   version     : str
   num_frames  : int
   shape       : int or (int,int)
   label       : str

are general parameters. They tell with which version this set 
As walking the data tree and extracting the data from the *hdf5* file 
is a bit cumbersome with h5py, there are a few convenience function in the 
:py:mod:`ptypy.io.h5rw` module.
      
Loading a data file
-------------------

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
