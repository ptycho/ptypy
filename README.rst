Phase Focus Limited of Sheffield, UK, has an international portfolio
of patents and pending applications which relate to ptychography.
A current list is available `here <http://www.phasefocus.com/patents/>`__.

Phase Focus grants royalty free licences of its patent rights for
non-commercial academic research use, for reconstruction of simulated
data and for reconstruction of data obtained at synchrotrons at X-ray
wavelengths. These licenses can be applied for online by
clicking on `this link <http://www.phasefocus.com/licence/>`__.

Phase Focus asserts that the software we have made available for
download may be capable of being used in circumstances which may
fall within the claims of one or more of the Phase Focus patents.
Phase Focus advises that you apply for a licence from it before
downloading any software from this website.

----

PtyPy - Ptychography Reconstruction for Python
==============================================

|ptypysite|

.. image:: https://github.com/ptycho/ptypy/workflows/ptypy%20tests/badge.svg?branch=master
    :target: https://github.com/ptycho/ptypy/actions

Welcome Ptychonaut!
-------------------
     
|ptypy| [#pronounciation]_ [#ptypypaper]_ is a
framework for scientific ptychography compiled by 
P. Thibault and B. Enders and other authors (see AUTHORS).

It is the result of years of experience in the field of ptychography condensed
into a versatile python package. The package covers the whole path of
ptychographic analysis after the actual experiment is completed
- from data management to reconstruction to visualization.

The main idea of ptypy is: *"Flexibility and Scalabality through abstraction"*. 
Most often, you will find a class for every concept of ptychography in 
|ptypy|. Using these or other more abstract base classes, new ideas
may be developed in a rapid manner without the cumbersome overhead of 
data management, memory access or
distributed computing. Additionally, |ptypy|
provides a rich set of utilities and helper functions,
especially for input and output

To get started quickly, please find the official documentation on
`the project pages <http://ptycho.github.io/ptypy>`__
or have a look at the examples in the ``templates`` directory.

Features
--------

* **Difference Map** [#dm]_ algorithm engine with power bound constraint [#power]_.
* **Maximum Likelihood** [#ml]_ engine with preconditioners and regularizers.
* A few more engines (RAAR, sDR, ePIE, ...).

* **Fully parallelized** using the Massage Passing Interface
  (`MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_). 
  Simply execute your script with::
  
    $ mpiexec -n [nodes] python <your_ptypy_script>.py

* **GPU acceleration** based on custom kernels, pycuda, and reikna.

* A **client-server** approach for visualization and control based on 
  `ZeroMQ <http://www.zeromq.org>`_ .
  The reconstruction may run on a remote hpc cluster while your desktop
  computer displays the reconstruction progress.

* **Mixed-state** reconstructions of probe and object [#states]_ for 
  overcoming partial coherence or related phenomena.
  
* **On-the-fly** reconstructions (while data is being acquired) using the
  the `PtyScan <http://http://ptycho.github.io/ptypy/rst/ptypy.core.html#ptypy.core.data.PtyScan>`_
  class in the `linking mode <http://ptycho.github.io/ptypy/rst/data_management.html#case-flyscan>`_


Installation
------------

Installation should be as simple as ::

   $ pip install .

or, as a user, ::

   $ pip install . --user


Dependencies
------------

Ptypy depends on standard python packages:
 * numpy
 * scipy
 * h5py
 * matplotlib & pillow (optional - required for plotting)
 * mpi4py (optional - required for parallel computing)
 * pyzmq (optional - required for the plotting client)
 
 
Quicklinks
----------
* | The complete `documentation <http://ptycho.github.io/ptypy/content.html#contents>`_ .

* | Starting from a **clean slate**?
  | Check out the `installation instructions <http://ptycho.github.io/ptypy/rst/getting_started.html#installation>`_ . 
  
* | You want to understand the **inner principles** of ptypy without 
    having to browse the source code?
  | Have a look at the `tutorials about its special classes <http://ptycho.github.io/ptypy/rst/concept.html#concepts>`_ .
  
* | Only interested in |ptypy|'s **data file structure** and 
    **management**? Indulge yourself `here <http://ptycho.github.io/ptypy/rst/data_management.html#ptyd-file>`__ for the structure and `here <http://ptycho.github.io/ptypy/rst/data_management.html#ptypy-data>`__  for the concepts.


Contribute
----------

- Issue Tracker: `<http://github.com/ptycho/ptypy/issues>`_
- Source Code: `<http://github.com/ptycho/ptypy>`_

Support
-------

If you are having issues, please let us know.


.. |ptypy| replace:: PtyPy

.. |ptypysite| image:: https://ptycho.github.io/ptypy/_static/logo_100px.png
         :target: https://ptycho.github.io/ptypy/


References
----------

.. [#pronounciation] Pronounced *typy*, forget the *p*, as in psychology.

.. [#ptypypaper] B.Enders and P.Thibault, *Proc. R. Soc. A* **472**, `doi <http://doi.org/10.1098/rspa.2016.0640>`__

.. [#states] P.Thibault and A.Menzel, *Nature* **494**, 68 (2013), `doi <http://dx.doi.org/10.1038/nature11806>`__

.. [#dm] P.Thibault, M.Dierolf *et al.*, *Science* **321**, 7 (2009), `doi <http://dx.doi.org/10.1126/science.1158573>`__

.. [#ml] P.Thibault and M.Guizar-Sicairos, *New J. of Phys.* **14**, 6 (2012), `doi <http://dx.doi.org/10.1088/1367-2630/14/6/063004>`__

.. [#power] K.Giewekemeyer *et al.*, **PNAS 108**, 2 (2007), `suppl. material <https://www.pnas.org/doi/10.1073/pnas.0905846107#supplementary-materials>`__, `doi <https://doi.org/10.1073/pnas.0905846107>`__
