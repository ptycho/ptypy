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
   energy 4.2883e+02
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
For sure we set

::

   >>> DEFAULT.recipe = RECIPE


::

   >>> class NumpyScan(PtyScan):
   >>>     
   >>>     # We overwrite the DEFAULT with the new DEFAULT.
   >>>     DEFAULT = DEFAULT
   >>>     
   >>>     def __init__(self,pars=None, **kwargs):
   >>>         super(NumpyScan, self).__init__(p, **kwargs)

At this point of initialisation it would be good to read in
the geometric information we stored in ``geofilepath``. We write a 
tiny file parser.

::

   >>> def extract_geo(base_path):
   >>>     out = {}
   >>>     with open(base_path+'geometry.txt') as f:
   >>>         for line in f:
   >>>             key, value = line.strip().split()
   >>>             out[key]=eval(value)
   >>>     return out
   >>>     


::

   >>> print extract_geo(save_path)
   {'distance': 0.15, 'energy': 428.83, 'shape': 256, 'psize': 2.4e-05}
   

Similarly we would need the same for the positions file

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

And the test

::

   >>> files, pos = extract_pos(save_path)
   >>> print files[:2]
   ['ccd/diffraction_0000.npy', 'ccd/diffraction_0001.npy']
   
   >>> print pos[:2]
   [(0.0014658, 0.0020175), (0.0018532, 0.0016686)]
   
