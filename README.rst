PTYPY - Ptychography Reconstruction for Python
==============================================

|ptypysite|

.. image:: https://github.com/ptycho/ptypy/workflows/ptypy%20tests/badge.svg?branch=master
    :target: https://github.com/ptycho/ptypy/actions

Welcome Ptychonaut!
-------------------
     
|ptypy| [#pronounciation]_ [#ptypypaper]_ is a
framework for scientific ptychography compiled by 
P. Thibault and B. Enders and licensed under the GPLv2 license.

It is the result of 7 years of experience in the field of ptychography condensed
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

To get started quickly, please find the official documentation at the project pages
`here <http://ptycho.github.io/ptypy>`_ or have a look at the examples in the ``templates`` directory.

Features
--------

* **Difference Map** [#dm]_ algorithm engine with power bound constraint [#power]_.
* **Maximum Likelihood** [#ml]_ engine with preconditioners and regularizers.
* A few more engines (RAAR, sDR, ePIE, ...).

* **Fully parallelized** (CPU only) using the Massage Passing Interface 
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
  the `PtyScan <http://http://ptycho.github.io/ptypy/rst/ptypy.core.html#ptypy.core.data.PtyScan>`_ class in the linking mode `linking mode <http://ptycho.github.io/ptypy/rst/data_management.html#case-flyscan>`_ 


Installation
------------

Installation should be as simple as ::

   $ sudo pip install .

or, as a user, ::

   $ pip install . --user


Dependencies
------------

Ptypy depends on standard python packages:
 * numpy
 * scipy
 * matplotlib
 * h5py
 * mpi4py (optional - required for parallel computing)
 * zeromq (optional - required for the offline plotting client)
 
 
Quicklinks
----------
* | The complete `documentation <http://ptycho.github.io/ptypy/content.html#contents>`_ .

* | Starting from a **clean slate**?
  | Check out the `installation instructions <http://ptycho.github.io/ptypy/rst/getting_started.html#installation>`_ . 
  
* | You want to understand the **inner principles** of ptypy without 
    having to browse the source code?
  | Have a look at the `tutorials about its special classes <http://ptycho.github.io/ptypy/rst/concept.html#concepts>`_ .
  
* | Only interested in |ptypy|'s **data file structure** and 
    **management**? Indulge yourself `here <http://ptycho.github.io/ptypy/rst/data_management.html#ptyd-file>`_ for the structure and `here <http://ptycho.github.io/ptypy/rst/data_management.html#ptypy-data>`_  for the concepts.


Contribute
----------

- Issue Tracker: `<http://github.com/ptycho/ptypy/issues>`_
- Source Code: `<http://github.com/ptycho/ptypy>`_

Support
-------

If you are having issues, please let us know.


License
-------

The project is licensed under a GPLv2 license.


.. |ptypy| replace:: PtyPy

.. |ptypysite| image:: https://ptycho.github.io/ptypy/_static/logo_100px.png
         :target: https://ptycho.github.io/ptypy/


References
----------

.. [#pronounciation] Pronounced *typy*, forget the *p*, as in psychology.

.. [#ptypypaper] B.Enders and P.Thibault, *Proc. R. Soc. A* **472**, `doi <http://doi.org/10.1098/rspa.2016.0640>`_

.. [#states] P.Thibault and A.Menzel, *Nature* **494**, 68 (2013), `doi <http://dx.doi.org/10.1038/nature11806>`_

.. [#dm] P.Thibault, M.Dierolf *et al.*, *Science* **321**, 7 (2009), `doi <http://dx.doi.org/10.1126/science.1158573>`_

.. [#ml] P.Thibault and M.Guizar-Sicairos, *New J. of Phys.* **14**, 6 (2012), `doi <http://dx.doi.org/10.1088/1367-2630/14/6/063004>`_

.. [#power] K.Giewekemeyer *et al.*, **PNAS 108**, 2 (2007), `suppl. material <https://www.pnas.org/doi/10.1073/pnas.0905846107#supplementary-materials>`__, `doi <https://doi.org/10.1073/pnas.0905846107>`__
