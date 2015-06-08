**************************
Getting started with Ptypy
**************************

Installation
============

General installation instructions
---------------------------------

|ptypy|_ is a python package and depends on a number of other
packages. Once its dependecies are met, ptypy should work *out-of-the-box*.

.. note:: 
   |ptypy| is developed for Python 2.7.x and is currently incompatible
   with Python 3.x. Python 3 support is planned in future releases.
          

Essential packages
^^^^^^^^^^^^^^^^^^

* `NumPy <https://pypi.python.org/pypi/numpy/1.9.2>`_ 
  (homepage: `<http://www.numpy.org>`_)
* `SciPy <https://pypi.python.org/pypi/scipy/0.15.1>`_ 
  (homepage: `<http://www.scipy.org>`_)
* `h5py <https://pypi.python.org/pypi/scipy/2.5.0>`_ 
  (homepage: `<http://www.h5py.org>`_)

Recommended packages for additional functionality
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Please note that |ptypy| as alpha release still lacks rigorous import 
checking. There may be parts of code that implicitly asks for any of the
packages listed here. 

* `Matplotlib <https://pypi.python.org/pypi/matplotlib/1.4.3>`_ 
  for any kind of plotting or image generation
  (homepage: `<http://www.matplotlib.org>`_) and python bindings for
  `QT4 <https://pypi.python.org/pypi/PyQt4/4.11.3>`_

* `mpi4py <https://pypi.python.org/pypi/mpi4py/1.3.1>`_ 
  for CPU-node parallelization, python bindings for the
  `Message Passaging Interface <http://www.mcs.anl.gov/research/projects/mpi/>`_, 
  (homepage: `<https://bitbucket.org/mpi4py/mpi4py/>`_)

* `pyzmq <https://pypi.python.org/pypi/pyzmq/14.6.0>`_ 
  for a threaded server on the main node in threaded asynchronous 
  client-server communication, python bindings for the
  ZeroMQ protocol, 
  (homepage: `<http://github.com/zeromq/pyzmq>`_)
  This package is needed for non-blocking plots of the reconstruction run.

Optional packages
^^^^^^^^^^^^^^^^^
Other very useful packages are

* `Ipython <https://pypi.python.org/pypi/ipython/3.1.0>`_ 
  (homepage: `<http://www.ipython.org>`_)

Installation on Debian/Ubuntu
-----------------------------
|Ptypy| is developed on the current LTS version of Ubuntu and installation
is straightforward.

Prerequisites
^^^^^^^^^^^^^
For Debian/Ubuntu, many of the required python packages are
available in the repositories. Just type (with sudo rights)
::
   $ sudo apt-get install python-numpy python-scipy python-h5py python-matplotlib python-mpi4py python-pyzmq

Get ptypy
^^^^^^^^^
Download and unpack or clone |ptypy|_ from the sources.
::
   $ wget https://github.com/ptycho/ptypy/archive/master.zip
   $ unzip master.zip -d /tmp/; rm master.zip

or make a clone of the |ptypy|_ github repository::
  
   $ git clone https://github.com/ptycho/ptypy.git /tmp/ptypy-master

CD into the download directory (e.g. /tmp/ptypy-master) and install |ptypy| with 
::
   $ python setup.py install --user 

The local --user install is generally recommended over the system wide
sudo install, such that you are able to quickly apply fixes yourself.
However, for an all-user system-wide install, type
::
   $ sudo python setup.py install 

Installation on Windows
-----------------------

This installation instruction were contributed by M. Stockmar and tested
with *Windows 8.1 Enterprise (64 bit)* on a core i7 Thinkpad w520 
in February 2015. 
No python was installed before on this machine. 

.. note::
   There might be also simpler ways to get a full scientific python 
   suite for 32 bit Windows. Please let us know if you managed to get
   |ptypy|_ running on a system not listed here.

Prerequisites
^^^^^^^^^^^^^

* Download and install python 2.7.x from `<http://www.python.org>`_ 
  (Make sure you have the 64 bit version)
  Click *yes* that you want add python to the path environment variable.

* Go to the command line and install wheel::
  
    $ pip install wheel

* Install Microsofts implementation for MPI to use multiple CPUs from
  `<https://msdn.microsoft.com/en-us/library/bb524831(v=vs.85).aspx>`_
  Other MPI implementation may work as well but were not tested.

* Install the QT-Framework if you don't have it already.
  `<http://qt-project.org/>`_


Download all other binaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The binaries will be downloaded from a `website <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_ 
which offers varies 
builds for different python version for Windows (both 32 and 64 bit) in
the form of a `python wheel <https://pypi.python.org/pypi/wheel>`_.
Make sure you choose the correct version for your windows and system. 

Numpy and some other builds are linked against the 
Intel Math Kernel Library (MKL) which is supposed to be fast.

* First make sure you have the Microsoft Visual C++ 2010 redistributable package (maybe also the 2008 version).
  
  * 2010 (64 bit, for 32 bit version check the website):
    `<http://www.microsoft.com/en-us/download/details.aspx?id=14632>`_ 
  * 2008 (64 bit, for 32 bit check the website):
    `<http://www.microsoft.com/en-us/download/details.aspx?id=15336>`_

* Now, go to to `<http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_
  and download the latest pip binaries and update pip::
    
    $ python.exe pip-6.0.8-py2.py3-none‑any.whl/pip install pip-6.0.8-py2.py3-none‑any.whl.
  
  (The file name might change slightly if a newer version of pip is available)

* Then download all the other binaries as whl-files.
  A whl-file can be installed via command line according to::
  
    $ pip install filename.whl
  
  Downloaded and installed the binaries in the following order:

  #. ``numpy``
  #. ``pillow`` (replacement for PIL)
  #. ``matplotlib``
  #. ``ipython``
  #. ``h5py``
  #. ``scipy`` (may install additional packages as well)
  #. ``mpi4py`` (choose the one for MS MPI if you have installed MS MPI)
  #. ``pyzmq``
  #. ``pyqt4`` (QT framework was installed before on the testing machine)
  #. ``pyside`` 

..
   pip install pyqt4  (QT framework was installed before on the testing machine)
   pip install pyside

   I hope I have not forgot anything but the website by C. Gohlke has everything you need.

Get ptypy
^^^^^^^^^
Download |ptypy| from `<https://github.com/ptycho/ptypy/archive/master.zip>`_
and unzip to any directory, for example ``C:\Temp``. 
Change into that directly and install from commandline::

  $ cd C:\Temp\ptypy-master
  $ python setup.py install


Quickstart with a minimal script
================================

.. include:: minimal_script.rst

Analysis of the generated files
-------------------------------

|ptypy| provides a few scripts to make life easier for the user. For 
example there is an automatic plotting script, that install on Unix
It has the syntax
::
   $ ptypy.plot <recon_file.ptyr> <template> <save_file.png>

In our case that translates to
::
   $ ptypy.plot /tmp/ptypy/recons/minimal/minimal_DM.ptyr default minimal.png

and the image looks like this (:numref:`minimal_result`)

.. figure:: ../img/minimal_result.png
   :width: 90 %
   :figclass: highlights
   :name: minimal_result
   
   Example plot made with ``ptypy.plot``

