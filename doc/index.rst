Welcome Ptychonaut!
===================
     
|ptypy| [#pronounciation]_ is a
framework for scientific ptychography compiled by 
P.Thibault and B. Enders and licensed under the GPLv2 license.

It comprises 7 years of experience in the field of ptychography condensed  
to a veratile python package. The package covers the whole path of 
ptychographic analysis after the actual experiment 
- from data management to reconstruction to visualization.

The main idea of ptypy is: *"Flexibility and Scalabality through abstraction"*. 
Most often, you will find a class for every concept of ptychography in 
|ptypy|. Using these or other more abstract base classes, new ideas
may be developed in a rapid manner without the cumbersome overhead of 
:py:mod:`data<ptypy.core.data>` management 
, memory access or :py:mod:`distributed <ptypy.utils.parallel>` computing. Additionally, |ptypy|
provides a rich set of :py:mod:`utilities <ptypy.utils>` and helper functions,
especially for :py:mod:`input/output <ptypy.io>`

Get started quickly :ref:`here <getting_started>` or with one of the examples in the ``templates`` directory.


Highlights
----------

* **Difference Map** [#dm]_ algorithm engine with power bound constraint
* **Maximum Likelihood** [#ml]_ engine with preconditioners and regularizers.

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
* | The complete :ref:`documentation <contents>`.

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

.. [#pronounciation] Pronounced *typy*, forget the *p*, as in psychology.

.. [#states] P.Thibault and A.Menzel, **Nature** 494, 68 (2013), `doi <http://dx.doi.org/10.1038/nature11806>`_

.. [#ml] P.Thibault and M.Guizar-Sicairos, **New J. of Phys.** 14, 6 (2012), `doi <http://dx.doi.org/10.1126/science.1158573>`_

.. [#dm] P.Thibault, M.Dierolf *et al.*, **New J. of Phys. 14**, 6 (2012), `doi <http://dx.doi.org/10.1088/1367-2630/14/6/063004>`_
