How to write an engine yourself
===============================

In this tutorial we want to to provide the ptypy user with the information
needed to create an engine compatible with the state mixture
expansion of ptychogrpahy as desribed in Thibault et. al 2012 [Thi2012]_ .

First we import ptypy and the utility module

::

   >>> import ptypy
   >>> from ptypy import utils as u
   >>> import numpy as np

Next we need to create a most basic input paramater tree. While there 
are many default values, we manually specify a more verbose output
and single precision.

::

   >>> p = u.Param()
   >>> p.verbose_level = 3
   >>> p.data_type = "single"

Now we need to create a set of scans that we wish to reconstruct 
in a single run. We will use a single scan that we call 'MF' and
marking the data source as 'test' to use the Ptypy internal 
:any:`MoonFlowerScan`

::

   >>> p.scans = u.Param()
   >>> p.scans.MF = u.Param()
   >>> p.scans.MF.data= u.Param()
   >>> p.scans.MF.data.source = 'test'
   >>> p.scans.MF.data.shape = 128
   >>> p.scans.MF.data.num_frames = 400

This bare parameter tree will be the input for the :any:`Ptycho`
class which is constructed at level 2, which means that it creates
all necessary basic :any:`Container` instances like *probe*, *object* 
*diff* , etc. It also loads the first chunk of data and creates all 
:any:`View` and :any:`POD` instances, as the verbose output will tell.

::

   >>> P = ptypy.core.Ptycho(p,level=2)
   Verbosity set to 3
   Data type:               single
   
   ---- Ptycho init level 1 -------------------------------------------------------
   Model: sharing probe between scans (one new probe every 1 scan)
   Model: sharing probe between scans (one new probe every 1 scan)
   
   ---- Ptycho init level 2 -------------------------------------------------------
   Prepared 310 positions
   Processing new data.
   
   ---- Analysis of the "common" arrays -------------------------------------------
   Check for weight or mask,  "weight2d"  .... True : shape = (128, 128)
   Check for positions, "positions_scan" .... True : shape = (310, 2)
   Scanning positions (310) are fewer than the desired number of scan points (400).
   Resetting `num_frames` to lower value
   ---- Analysis done -------------------------------------------------------------
   
   ROI center is [ 64.  64.], automatic guess is [ 63.55483871  63.56774194].
   Feeding data chunk
   Importing data from MF as scan MF.
   End of scan reached
   End of scan reached
   
   --- Scan MF photon report ---
   Total photons   : 1.15e+10 
   Average photons : 3.72e+07
   Maximum photons : 7.86e+07
   -----------------------------
   
   ---- Creating PODS -------------------------------------------------------------
   Found these probes : 
   Found these objects: 
   Process 0 created 310 new PODs, 1 new probes and 1 new objects.
   
   ---- Probe initialization ------------------------------------------------------
   Initializing probe storage S00G00 using scan MF
   Found no photon count for probe in parameters.
   Using photon count 7.86e+07 from photon report
   
   ---- Object initialization -----------------------------------------------------
   Initializing object storage S00G00 using scan MF
   Simulation resource is object transmission
   
   ---- Creating exit waves -------------------------------------------------------
   
   Process #0 ---- Total Pods 310 (310 active) ----
   --------------------------------------------------------------------------------
   (C)ontnr : Memory : Shape            : Pixel size      : Dimensions      : Views
   (S)torgs : (MB)   : (Pixel)          : (meters)        : (meters)        : act. 
   --------------------------------------------------------------------------------
   Cprobe   :    0.1 : complex64
   S00G00   :    0.1 :        1*128*128 :   6.36*6.36e-08 :   8.14*8.14e-06 :   310
   Cmask    :    5.1 :   bool
   S0000    :    5.1 :      310*128*128 :   1.72*1.72e-04 :   2.20*2.20e-02 :   310
   Cexit    :   40.6 : complex64
   S0000G00 :   40.6 :      310*128*128 :   6.36*6.36e-08 :   8.14*8.14e-06 :   310
   Cobj     :    3.5 : complex64
   S00G00   :    3.5 :        1*664*652 :   6.36*6.36e-08 :   4.22*4.15e-05 :   310
   Cdiff    :   20.3 : float32
   S0000    :   20.3 :      310*128*128 :   1.72*1.72e-04 :   2.20*2.20e-02 :   310
   
   
   

A quick look at the diffraction data

::

   >>> fig = u.plot_storage(P.diff.S['S0000'],0,slices=(slice(2),slice(None),slice(None)),modulus='log')
See :numref:`ownengine_00` for the plotted image.

.. figure:: ../_img/ownengine_00.png
   :width: 70 %
   :figclass: highlights
   :name: ownengine_00

   Plot of simulated diffraction data for the first two positions.

Probe and object are not so exciting to look at for now. As default,
probes are initialized with an aperture like support.

::

   >>> fig = u.plot_storage(P.probe.S['S00G00'],1)
See :numref:`ownengine_01` for the plotted image.

.. figure:: ../_img/ownengine_01.png
   :width: 70 %
   :figclass: highlights
   :name: ownengine_01

   Plot of the starting guess for the probe.

Now we can start implementing a simple DM algorithm. We need three basic
functions, one is the ``fourier_update`` that implements the Fourier
modulus constraint.

.. math::
   \psi_{d,\lambda,k} = \mathcal{D}_{\lambda,z}^{-1}\left\{\sqrt{I_{d}}\frac{\mathcal{D}_{\lambda,z} \{\psi_{d,\lambda,k}\}}{\sum_{\lambda,k} |\mathcal{D}_{\lambda,z} \{\psi_{d,\lambda,k}\} |^2}\right\}


::

   >>> def fourier_update(pods):
   >>>     import numpy as np
   >>>     pod = pods.values()[0]
   >>>     # Get Magnitude and Mask
   >>>     mask = pod.mask
   >>>     modulus = np.sqrt(np.abs(pod.diff))
   >>>     # Create temporary buffers
   >>>     Imodel= np.zeros_like(pod.diff) 
   >>>     err = 0.                             
   >>>     Dphi = {}                                
   >>>     # Propagate the exit waves
   >>>     for gamma, pod in pods.iteritems():
   >>>         Dphi[gamma]= pod.fw( 2*pod.probe*pod.object - pod.exit )
   >>>         Imodel += Dphi[gamma] * Dphi[gamma].conj()
   >>>     # Calculate common correction factor
   >>>     factor = (1-mask) + mask* modulus /(np.sqrt(Imodel) + 1e-10)
   >>>     # Apply correction and propagate back
   >>>     for gamma, pod in pods.iteritems():
   >>>         df = pod.bw(factor*Dphi[gamma])-pod.probe*pod.object
   >>>         pod.exit += df
   >>>         err += np.mean(np.abs(df*df.conj()))
   >>>     # Return difference map error on exit waves
   >>>     return err


::

   >>> def probe_update(probe,norm,pods,fill=0.):
   >>>     """
   >>>     Updates `probe`. A portion `fill` of the probe is kept from 
   >>>     iteration to iteration. Requires `norm` buffer and pod dictionary
   >>>     """
   >>>     probe *= fill
   >>>     norm << fill + 1e-10
   >>>     for name,pod in pods.iteritems():
   >>>         if not pod.active: continue
   >>>         probe[pod.pr_view] += pod.object.conj() * pod.exit
   >>>         norm[pod.pr_view] += pod.object * pod.object.conj()
   >>>     # For parallel usage (MPI) we have to communicate the buffer arrays
   >>>     probe.allreduce()
   >>>     norm.allreduce()
   >>>     probe /= norm


::

   >>> def object_update(obj,norm,pods,fill=0.):
   >>>     """
   >>>     Updates `object`. A portion `fill` of the object is kept from 
   >>>     iteration to iteration. Requires `norm` buffer and pod dictionary
   >>>     """
   >>>     obj *= fill
   >>>     norm << fill + 1e-10
   >>>     for pod in pods.itervalues():
   >>>         if not pod.active: continue
   >>>         pod.object += pod.probe.conj() * pod.exit
   >>>         norm[pod.ob_view] += pod.probe * pod.probe.conj()
   >>>     obj.allreduce()
   >>>     norm.allreduce()
   >>>     obj /= norm


::

   >>> def iterate(Ptycho, num):
   >>>     # generate container copies
   >>>     obj_norm = P.obj.copy(fill=0.)
   >>>     probe_norm = P.probe.copy(fill=0.)
   >>>     errors = []
   >>>     for i in range(num):
   >>>         err = 0
   >>>         # fourier update
   >>>         for di_view in Ptycho.diff.V.itervalues():
   >>>             if not di_view.active: continue
   >>>             err += fourier_update(di_view.pods)
   >>>         # probe update
   >>>         probe_update(Ptycho.probe, probe_norm, Ptycho.pods)
   >>>         # object update
   >>>         object_update(Ptycho.obj, obj_norm, Ptycho.pods)
   >>>         # print error
   >>>         errors.append(err)
   >>>         if i % 3==0: print err
   >>>     # cleanup
   >>>     P.obj.delete_copy()
   >>>     P.probe.delete_copy()
   >>>     #return error
   >>>     return errors

We start of with a small number of iterations.

::

   >>> iterate(P,9)
   438128.131012
   315525.085239
   238082.913346
   

We note that the error (here only displayed for 3 iterations) is 
already declining. That is a good sign. 
Let us have a look how the probe has developed.

::

   >>> fig = u.plot_storage(P.probe.S['S00G00'],2)
See :numref:`ownengine_02` for the plotted image.

.. figure:: ../_img/ownengine_02.png
   :width: 70 %
   :figclass: highlights
   :name: ownengine_02

   Plot of the reconstructed probe after 9 iterations. We observe that
   the actaul illumination of the sample must be larger than the initial
   guess.

Looks like the probe is on a good way. How about the object?

::

   >>> fig = u.plot_storage(P.obj.S['S00G00'],3,slices=(slice(1),slice(120,-120),slice(120,-120)))
See :numref:`ownengine_03` for the plotted image.

.. figure:: ../_img/ownengine_03.png
   :width: 70 %
   :figclass: highlights
   :name: ownengine_03

   Plot of the reconstructed obejct after 9 iterations. It is not quite
   clear what object is reconstructed

Ok, let us do some more iterations. 36 will do.

::

   >>> iterate(P,36)
   158772.734063
   92001.2507792
   61221.5036771
   48127.1963039
   38992.4460977
   27961.8069756
   22375.8444046
   19797.2539501
   18647.1967747
   17682.6146338
   17755.6180473
   17796.2601212
   

Error is still on a steady descent. Let us look at the final 
reconstructed probe and object.

::

   >>> fig = u.plot_storage(P.probe.S['S00G00'],4)
See :numref:`ownengine_04` for the plotted image.

.. figure:: ../_img/ownengine_04.png
   :width: 70 %
   :figclass: highlights
   :name: ownengine_04

   Plot of the reconstructed probe after a total of 45 iterations.
   It's a moon !


   >>> fig = u.plot_storage(P.obj.S['S00G00'],5,slices=(slice(1),slice(120,-120),slice(120,-120)))
See :numref:`ownengine_05` for the plotted image.

.. figure:: ../_img/ownengine_05.png
   :width: 70 %
   :figclass: highlights
   :name: ownengine_05

   Plot of the reconstructed object after a total of 45 iterations.
   It's a bunch of flowers !


.. [Thi2012] P. Thibault and A. Menzel, **Nature** 494, 68 (2013)


