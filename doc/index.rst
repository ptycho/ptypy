Welcome Ptychonaut!
===================
     
|ptypy| [#pronounciation]_ is a
framework for scientific ptychography compiled by 
P.Thibault and B. Enders amd licensed under the GPLv2 license.

It comprises 7 years of experience in the field of ptychography condensed  
to a veratile python package. The package covers the whole path of 
ptychographic analysis after the actual experiment 
- from data management to reconstruction to visualization.

The main idea of ptypy is: *"Flexibility and Scalabality through abstraction"*. 
Most often, you will find a class for every concept of ptychography in 
|ptypy|. Using these or other more abstract base classes, new ideas
may be developed in a rapid manner without the cumbersome overhead of data 
management, memory access or distributed computing.


To get started quickly, look at the examples in the template directory. You will
also need to prepare your data in a hdf5 file and following a structure that
ptypy can understand. Ptypy provides already routines to prepare data from three
beamlines (cSAXS, PSI; I13, Diamond; and I22, ESRF) and more will come.

Highlights
----------

* **Difference Map** algorithm engine with power bound constraint
* **Maximum Likelihood** engine with preconditioners and regularizers.

* **Fully parallelized** (CPU only) using the Massage Passing Interface 
  (`MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_). 
  Simply execute your script with::
  
    $ mpiexec -n [nodes] python <your_ptypy_script>.py

* A **client-server** approach for visualization and control based on 
  `ZeroMQ <http://www.zeromq.org>`_ .
  The reconstruction may run on a remote hpc cluster while your desktop
  computer displays the reconstruction progress.
  

* **Mixed-state** reconstructions of probe and object [#states]_ for 
  overcoming partial coherence or related phenomane.
  
* **On-the-fly** reconstructions (while data is being acquired) using the
  the :any:`PtyScan` class in the linking mode :ref:`linking mode<case_flyscan>` 


Quicklinks
----------

* | Starting from a **clean slate**?
  | Check out the :ref:`installation instructions <installation>` 
  
* | You want to understand the **inner principles** of ptypy without 
    having to browse the source code?
  | Have a look at the :ref:`tutorials about its special classes <concepts>`.
  
* | Only interested in |ptypy|'s **data file structure** and 
    **management**?
  | Indulge yourself :ref:`here<ptyd_file>` for the structure and 
    :ref:`here<ptypy_data>` for the concepts.




.. rubric:: Footnotes

.. [#pronounciation] Pronounced typy, forget the p, as in ptychography or psychology.
.. [#states] P. Thibault and A. Menzel, **Nature** 494, 68 (2013)
