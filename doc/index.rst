Welcome Ptychonaut!
===================
     
|ptypy| [#Enders2016]_ is a
framework for scientific ptychography compiled by 
P.Thibault, B. Enders, and others (see AUTHORS).

It is the result of years of experience in the field of ptychography condensed
into a versatile python package. The package covers the whole path of
ptychographic analysis after the actual experiment is completed
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

* **Difference Map** [#dm]_ algorithm engine with power bound constraint [#power]_.
* **Maximum Likelihood** [#ml]_ engine with preconditioners and regularizers.
* A few more engines (RAAR, sDR, ePIE, ...).

* **Fully parallelized** using the Massage Passing Interface
  (`MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_). 
  Simply execute your script with::
  
    $ mpiexec/mpirun -n [nodes] python <your_ptypy_script>.py

* **GPU acceleration** based on custom kernels, pycuda, and reikna.

* A **client-server** approach for visualization and control based on 
  `ZeroMQ <http://www.zeromq.org>`_ .
  The reconstruction may run on a remote hpc cluster while your desktop
  computer displays the reconstruction progress.
  

* **Mixed-state** reconstructions of probe and object [#Thi2013]_ for
  overcoming partial coherence or related phenomena.
  
* **On-the-fly** reconstructions (while data is being acquired) using the
  the :py:class:`PtyScan` class in the linking mode :ref:`linking mode<case_flyscan>` 


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

.. [#Enders2016] B.Enders and P.Thibault, **Proc. R. Soc. A** 472, 20160640 (2016), `doi <http://dx.doi.org/10.1098/rspa.2016.0640>`__

.. [#Thi2013] P.Thibault and A.Menzel, **Nature** 494, 68 (2013), `doi <http://dx.doi.org/10.1038/nature11806>`__

.. [#ml] P.Thibault and M.Guizar-Sicairos, **New J. of Phys. 14**, 6 (2012), `doi <http://dx.doi.org/10.1088/1367-2630/14/6/063004>`__

.. [#dm] P.Thibault, M.Dierolf *et al.*, **Ultramicroscopy 109**, 4 (2009), `doi <https://doi.org/10.1016/j.ultramic.2008.12.011>`__

.. [#power] K.Giewekemeyer *et al.*, **PNAS 108**, 2 (2007), `suppl. material <https://www.pnas.org/doi/10.1073/pnas.0905846107#supplementary-materials>`__, `doi <https://doi.org/10.1073/pnas.0905846107>`__

