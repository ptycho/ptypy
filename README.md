PTYPY - Ptychography Reconstruction for Python
==============================================

[![DOI](https://zenodo.org/badge/6834/ptycho/ptypy.png)](http://dx.doi.org/10.5281/zenodo.12480)

Ptypy (pronounced typy, forget the p, as in ptychography or psychology) is a
reconstruction package for ptychographic datasets.

To get started quickly, look at the examples in the template directory. You will
also need to prepare your data in a hdf5 file and following a structure that
ptypy can understand. Ptypy provides already routines to prepare data from three
beamlines (cSAXS, PSI; I13, Diamond; and I22, ESRF) and more will come.

Features
--------

Ptypy was designed with flexibility in mind: it should allow rapid development
of new ideas. To this end, much of the "ugly" details have been hidden in
advanced containers that manage data and access "views" onto them. 

Currently implemented:

* Fully parallelized (using MPI)
* Difference map algorithm with power bound constraint
* Maximum Likelihood with preconditioners and regularizers.
* Mixed-state reconstructions of probe and object
* On-the-fly reconstructions (while data is being acquired) 

Installation
------------

Installation should be as simple as
::
   $ sudo python setup.py install

or, as a user,
::
   $ python setup.py install --user

Dependencies
------------

Ptypy depends on standard python packages:
 * numpy
 * scipy
 * matplotlib
 * h5py
 * mpi4py (optional - required for parallel computing)
 * zeromq (optional - required for the offline plotting client)
 * maybe some more we forgot to put in this list.

Contribute
----------

- Issue Tracker: github.com/ptycho/ptypy/issues
- Source Code: github.com/ptycho/ptypy

Support
-------

If you are having issues, please let us know.

License
-------

The project is licensed under a GPLv2 license.
