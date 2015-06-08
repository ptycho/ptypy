.. _parameters:

*************************
Ptypy parameter structure
*************************

.. py:data:: .verbose_level(int)

   *(0)* Verbosity level

   Verbosity level for information logging.
    - ``0``: Only errors
    - ``1``: Warning
    - ``2``: Process Information
    - ``3``: Object Information
    - ``4``: Debug

   *default* = ``1 (>0, <3)``

.. py:data:: .data_type(str)

   *(1)* Reconstruction floating number precision

   Reconstruction floating number precision (``'single'`` or ``'double'``)

   *default* = ``single``

.. py:data:: .run(str)

   *(2)* Reconstruction identifier

   Reconstruction run identifier. If ``None``, the run name will be constructed at run time from other information.

   *default* = ``None``

.. py:data:: .dry_run(bool)

   *(3)* Dry run switch 

   Run everything skipping all memory and cpu-heavy steps (and file saving).
   **NOT IMPLEMENTED**

   *default* = ``FALSE``


.io
===

.. py:data:: .io(Param)

   *(4)* Global parameters for I/O

   

   *default* = ``None``

.. py:data:: .io.home(dir)

   *(5)* Base directory for all I/O

   home is the root directory for all input/output operations. All other path parameters that are relative paths will be relative to this directory.

   *default* = ``./``

.. py:data:: .io.rfile(str)

   *(6)* Reconstruction file name (or format string)

   Reconstruction file name or format string (constructed against runtime dictionary)

   *default* = ``recons/%(run)s/%(run)s_%(engine)s_%(iterations)04d.ptyr``


.io.autosave
------------

.. py:data:: .io.autosave(Param)

   *(7)* Auto-save options

   

   *default* = ``None``

.. py:data:: .io.autosave.active(bool)

   *(8)* Activation switch

   If ``True`` the current reconstruction will be saved at regular intervals. **unused**

   *default* = ``TRUE``

.. py:data:: .io.autosave.interval(int)

   *(9)* Auto-save interval

   If ``>0`` the current reconstruction will be saved at regular intervals according to the pattern in :py:data:`paths.autosave` . If ``<=0`` not automatic saving

   *default* = ``10 (>-1)``

.. py:data:: .io.autosave.rfile(str)

   *(10)* Auto-save file name (or format string)

   Auto-save file name or format string (constructed against runtime dictionary)

   *default* = ``dumps/%(run)s/%(run)s_%(engine)s_%(iterations)04d.ptyr``


.io.interaction
---------------

.. py:data:: .io.interaction(Param)

   *(11)* Server / Client parameters

   If ``None`` or ``False`` is passed here in script instead of a Param, it translates to  ``active=False`` i.e. no ZeroMQ interaction server. 

   *default* = ``None``

.. py:data:: .io.interaction.active(bool)

   *(12)* Activation switch

   Set to ``False`` for no  ZeroMQ interaction server
   

   *default* = ``TRUE``

.. py:data:: .io.interaction.primary_address(str)

   *(13)* The address the server is listening to.

   Wenn running ptypy on a remote server, it is the servers network address. 

   *default* = ``tcp://127.0.0.1``

.. py:data:: .io.interaction.primary_port(int)

   *(14)* The port the server is listening to.

   The port the server is listening to.

   *default* = ``5560``

.. py:data:: .io.interaction.port_range(str)

   *(15)* The port range opened to clients.

   The port range opened to clients.

   *default* = ``"5571:5571"``


.io.autoplot
------------

.. py:data:: .io.autoplot(Param)

   *(16)* Plotting client parameters

   In script you may set this parameter to ``None`` or ``False`` for no automatic plotting.
   

   *default* = ``None``

.. py:data:: .io.autoplot.imfile(str)

   *(17)* Plot images file name (or format string)

   

   *default* = ``plots/%(run)s/%(run)s_%(engine)s_%(iterations)04d.png``

.. py:data:: .io.autoplot.interval(int)

   *(18)* Number of iterations between plot updates

   Requests to the server will happen with this iteration intervals. Note that this will work only if interaction.polling_interval is smaller or equal to this number. If ``interval =0`` plotting is disabled which should be used, when ptypy is run on a cluster.

   *default* = ``1 (>-1)``

.. py:data:: .io.autoplot.threaded(bool)

   *(19)* Live plotting switch

   If ``True``, a plotting client will be spawned in a new thread and connected at initialization. If ``False``, the master node will carry out the plotting, pausing the reconstruction. This option should be set to False when ptypy is run on a cluster.

   *default* = ``TRUE``

.. py:data:: .io.autoplot.layout(str, Param)

   *(20)* Options for default plotter (not implemented yet)

   Options for default plotter (not implemented yet)

   *default* = ``None``

.. py:data:: .io.autoplot.dump(bool)

   *(21)* Switch to dump plots as image files

   

   *default* = ``TRUE``

.. py:data:: .io.autoplot.make_movie(bool)

   *(22)* Produce reconstruction movie after the reconstruction.

   Switch to request the production of a movie from the dumped plots at the end of the reconstruction.

   *default* = ``FALSE``


.scan
=====

.. py:data:: .scan(Param)

   *(23)* Scan parameters

   This categrogy specifies defaults for all scans. Scan-specific parameters are stored in scans.scan_%%

   *default* = ``None``

.. py:data:: .scan.tags(str)

   *(24)* Comma seperated string tags describing the data input

   [deprecated?]

   *default* = ``None``

.. py:data:: .scan.if_conflict_use_meta(bool)

   *(25)* Give priority to metadata relative to input parameters

   [useful?]

   *default* = ``TRUE``


.scan.data
----------

.. py:data:: .scan.data(Param)

   *(26)* Data preparation parameters

   

   *default* = ``None``

.. py:data:: .scan.data.recipe(ext)

   *(27)* Data preparation recipe container

   

   *default* = ``None``

.. py:data:: .scan.data.source(file)

   *(28)* Describes where to get the data from.


   Accepted values are:
    - ``'file'``: data will be read from a .ptyd file.
    - any valid recipe name: data will be prepared using the recipe.
    - ``'sim'`` : data will be simulated according to parameters in simulation  

   *default* = ``None``

.. py:data:: .scan.data.dfile(file)

   *(29)* Prepared data file path

   If source was ``None`` or ``'file'``, data will be loaded from this file and processing as well as saving is deactivated. If source is the name of an experiment recipe or path to a file, data will be saved to this file

   *default* = ``None``

.. py:data:: .scan.data.label(str)

   *(30)* The scan label

   Unique string identifying the scan

   *default* = ``None``

.. py:data:: .scan.data.shape(int, tuple)

   *(31)* Shape of the region of interest cropped from the raw data.

   Cropping dimension of the diffraction frame
   Can be None, (dimx, dimy), or dim. In the latter case shape will be (dim, dim).

   *default* = ``None``

.. py:data:: .scan.data.save(str)

   *(32)* Saving mode

   Mode to use to save data to file.
    - ``None``: No saving 
    - ``'merge'``: attemts to merge data in single chunk **[not implemented]**
    - ``'append'``: appends each chunk in master \*.ptyd file
    - ``'link'``: appends external links in master \*.ptyd file and stores chunks separately in the path given by the link. Links file paths are relative to master file.

   *default* = ``None``

.. py:data:: .scan.data.center(tuple)

   *(33)* Center (pixel) of the optical axes in raw data

   If ``None``, this parameter will be set by :py:data:`~.scan.data.auto_center` or elsewhere

   *default* = ``None``

.. py:data:: .scan.data.psize(float, tuple)

   *(34)* Detector pixel size

   Dimensions of the detector pixels (in meters)

   *default* = ``None (>0.0)``

.. py:data:: .scan.data.distance(float)

   *(35)* Sample-to-detector distance

   In meters.

   *default* = ``None (>0.0)``

.. py:data:: .scan.data.rebin(int)

   *(36)* Rebinning factor

   Rebinning factor for the raw data frames. ``'None'`` or ``1`` both mean *no binning*

   *default* = ``None (>1, <8)``

.. py:data:: .scan.data.orientation(int, tuple)

   *(37)* Data frame orientation

    - ``None`` or ``0``: correct orientation
    - ``1``: invert columns (numpy.flip_lr)
    - ``2``: invert columns, invert rows
    - ``3``: invert rows  (numpy.flip_ud)
    - ``4``: transpose (numpy.transpose)
    - ``4+i``: tranpose + other operations from above
   
   Alternatively, a 3-tuple of booleans may be provided ``(do_transpose, do_flipud, do_fliplr)``

   *default* = ``None``

.. py:data:: .scan.data.energy(float)

   *(38)* Photon energy of the incident radiation

   

   *default* = ``None (>0.0)``

.. py:data:: .scan.data.min_frames(int)

   *(39)* Minimum number of frames loaded by each node

   

   *default* = ``1``

.. py:data:: .scan.data.num_frames(int)

   *(40)* Maximum number of frames to be prepared

   If `positions_theory` are provided, num_frames will be ovverriden with the number of positions available

   *default* = ``None``

.. py:data:: .scan.data.chunk_format(str)

   *(41)* Appendix to saved files if save == 'link'

   

   *default* = ``.chunk%02d``

.. py:data:: .scan.data.auto_center(bool)

   *(42)* Determine if center in data is calculated automatically

    - ``False``, no automatic centering 
    - ``None``, only if :py:data:`center` is ``None`` 
    - ``True``, it will be enforced

   *default* = ``None``

.. py:data:: .scan.data.load_parallel(str)

   *(43)* Determines what will be loaded in parallel

   Choose from ``None``, ``'data'``, ``'common'``, ``'all'``

   *default* = ``data``

.. py:data:: .scan.data.positions_theory(ndarray)

   *(44)* Theoretical positions for this scan

   If provided, experimental positions from :any:`PtyScan` subclass will be ignored. If data preparation is called from Ptycho instance, the calculated positions from the :py:func:`ptypy.core.xy.from_pars` dict will be inserted here

   *default* = ``None``

.. py:data:: .scan.data.experimentID(str)

   *(45)* Name of the experiment

   If None, a default value will be provided by the recipe.

   *default* = ``None``

.. py:data:: .scan.data.simulation(Param)

   *(46)* Simulated data as a preparation

   Similar to scan, simulation takes Parameters trees in the same form `illumination`, `sample` and `xy`. Any item in these trees will take precedence over scan specific parameters in the simulated scan.

   *default* = ``None``

.. py:data:: .scan.data.simulation.detector(Param, str, NoneType)

   *(47)* Detector parameters

   Can also be ``None`` if no detector specific filter is wanted or a string that matches one of the templates in the detector module

   *default* = ``None``

.. py:data:: .scan.data.simulation.psf(float)

   *(48)* Gaussian point spread in detector

   Value passed here represents the FWHM of a Gaussian. ``None`` means no point spread.

   *default* = ``None``


.scan.sharing
-------------

.. py:data:: .scan.sharing(Param)

   *(49)* Scan sharing options

   

   *default* = ``None``

.. py:data:: .scan.sharing.object_share_with(str)

   *(50)* Label or index of scan to share object with.

   Possible values:
    - ``None``: Do not share
    - *(string)*: Label of the scan to share with
    - *(int)*: Index of scan to share with

   *default* = ``None``

.. py:data:: .scan.sharing.object_share_power(float)

   *(51)* Relative power for object sharing

   

   *default* = ``1 (>0.0)``

.. py:data:: .scan.sharing.probe_share_with(str)

   *(52)* Label or index of scan to share probe with.

   Possible values:
    - ``None``: Do not share
    - *(string)*: Label of the scan to share with
    - *(int)*: Index of scan to share with

   *default* = ``None``

.. py:data:: .scan.sharing.probe_share_power(float)

   *(53)* Relative power for probe sharing

   

   *default* = ``1 (>0.0)``


.scan.geometry
--------------

.. py:data:: .scan.geometry(Param)

   *(54)* Physical parameters

   All distances are in meters. Other units are specified in the documentation strings.

   *default* = ``None``

.. py:data:: .scan.geometry.energy(float)

   *(55)* Energy (in keV)

   If ``None``, uses `lam` instead.

   *default* = ``6.2 (>0.0)``

.. py:data:: .scan.geometry.lam(float)

   *(56)* Wavelength

   Used only if `energy` is ``None``

   *default* = ``None (>0.0)``

.. py:data:: .scan.geometry.distance(float)

   *(57)* Distance from object to detector

   

   *default* = ``7.19 (>0.0)``

.. py:data:: .scan.geometry.psize(float)

   *(58)* Pixel size in Detector plane

   

   *default* = ``0.000172 (>0.0)``

.. py:data:: .scan.geometry.resolution(float)

   *(59)* Pixel size in Sample plane

   This parameter is used only for simulations

   *default* = ``None (>0.0)``

.. py:data:: .scan.geometry.propagation(str)

   *(60)* Propagation type

   Either "farfield" or "nearfield"

   *default* = ``farfield``


.scan.xy
--------

.. py:data:: .scan.xy(Param)

   *(61)* Parameters for scan patterns

   These parameters are useful in two cases:
    - When the experimental positions are not known (no encoders)
    - When using the package to simulate data.
   
   In script an array of shape *(N,2)* may be passed here instead of a Param or dictionary as an **override**

   *default* = ``None``

.. py:data:: .scan.xy.model(str)

   *(62)* Scan pattern type

   The type must be one of the following:
    - ``None``: positions are read from data file.
    - ``'raster'``: raster grid pattern
    - ``'round'``: concentric circles pattern
    - ``'spiral'``: spiral pattern
   
   In script an array of shape *(N,2)* may be passed here instead

   *default* = ``None (>0.0)``

.. py:data:: .scan.xy.spacing(float, tuple)

   *(63)* Pattern spacing

   Spacing between scan positions. If the model supports asymmetric scans, a tuple passed here will be interpreted as *(dy,dx)* with *dx* as horizontal spacing and *dy* as vertical spacing. If ``None`` the value is calculated from `extent` and `steps`
   

   *default* = ``1.50E-06 (>0.0)``

.. py:data:: .scan.xy.steps(int, tuple)

   *(64)* Pattern step count

   Number of steps with length *spacing* in the grid. A tuple *(ny,nx)* provided here can be used for a different step in vertical ( *ny* ) and horizontal direction ( *nx* ). If ``None`` the, step count is calculated from `extent` and `spacing`

   *default* = ``10 (>0)``

.. py:data:: .scan.xy.extent(float, tuple)

   *(65)* Rectangular extent of pattern

   Defines the absolut maximum extent. If a tuple *(ly,lx)* is provided the extent may be rectangular rather than square. All positions outside of `extent` will be discarded. If ``None`` the extent will is `spacing` times `steps`

   *default* = ``1.50E-05 (>0.0)``

.. py:data:: .scan.xy.offset(float, tuple)

   *(66)* Offset of scan pattern relative to origin


   If tuple, the offset may differ in *x* and *y*. Please not that the offset will be included when removing scan points outside of `extend`.

   *default* = ``0``

.. py:data:: .scan.xy.jitter(float, tuple)

   *(67)* RMS of jitter on sample position

   **Only use in simulation**. Adds a random jitter to positions.

   *default* = ``0``

.. py:data:: .scan.xy.count(int)

   *(68)* Number of scan points


   Only return return positions up to number of `count`.

   *default* = ``None``


.scan.illumination
------------------

.. py:data:: .scan.illumination(Param)

   *(69)* Illumination model (probe)

   
   In script, you may pass directly a three dimensional  numpy.ndarray here instead of a `Param`. This array will be copied to the storage instance with no checking whatsoever. Used in `~ptypy.core.illumination`

   *default* = ``None (>0.0)``

.. py:data:: .scan.illumination.model(str)

   *(70)* Type of illumination model

   One of:
    - ``None`` : model initialitziation defaults to flat array filled with the specified number of photons
    - ``'recon'`` : load model from previous reconstruction, see `recon` Parameters
    - ``'stxm'`` : Estimate model from autocorrelation of mean diffraction data
    - *<resource>* : one of ptypys internal image resource strings
    - *<template>* : one of the templates inillumination module
   
   In script, you may pass a numpy.ndarray here directly as the model. It is considered as incoming wavefront and will be propagated according to `propagation` with an optional `aperture` applied before

   *default* = ``None``

.. py:data:: .scan.illumination.photons(int)

   *(71)* Number of photons in the incident illumination

   A value specified here will take precedence over calculated statistics from the loaded data.

   *default* = ``None (>0)``

.. py:data:: .scan.illumination.recon(Param)

   *(72)* Parameters to load from previous reconstruction

   

   *default* = ``None``

.. py:data:: .scan.illumination.recon.rfile(file)

   *(73)* Path to a ``.ptyr`` compatible file

   

   *default* = ``\*.ptyr``

.. py:data:: .scan.illumination.recon.ID(NoneType)

   *(74)* ID (label) of storage data to load

   ``None`` means any ID

   *default* = ``None``

.. py:data:: .scan.illumination.recon.layer(float)

   *(75)* Layer (mode) of storage data to load

   ``None`` means all layers, choose ``0`` for main mode

   *default* = ``None``

.. py:data:: .scan.illumination.stxm(Param)

   *(76)* Parameters to initialize illumination from diffraction data

   

   *default* = ``None``

.. py:data:: .scan.illumination.stxm.label(str)

   *(77)* Scan label of diffraction that is to be used for probe estimate

   ``None``, own scan label is used

   *default* = ``None``

.. py:data:: .scan.illumination.aperture(Param)

   *(78)* Beam aperture parameters

   

   *default* = ``None``

.. py:data:: .scan.illumination.aperture.form(str)

   *(79)* One of None, 'rect' or 'circ'

   One of:
    - ``None`` : no aperture, this may be useful for nearfield
    - ``'rect'`` : rectangular aperture
    - ``'circ'`` : circular aperture

   *default* = ``circ (>0.0)``

.. py:data:: .scan.illumination.aperture.diffuser(float)

   *(80)* Noise in the transparen part of the aperture

   Can be either:
    - ``None`` : no noise
    - ``2-tuple`` : noise in phase (amplitude (rms), minimum feature size)
    - ``4-tuple`` : noise in phase & modulus (rms, mfs, rms_mod, mfs_mod)

   *default* = ``None (>0.0)``

.. py:data:: .scan.illumination.aperture.size(float)

   *(81)* Aperture width or diameter

   May also be a tuple *(vertical,horizontal)* in case of an asymmetric aperture 

   *default* = ``None (>0.0)``

.. py:data:: .scan.illumination.aperture.edge(int)

   *(82)* Edge width of aperture (in pixels!)

   

   *default* = ``2 (>0)``

.. py:data:: .scan.illumination.aperture.central_stop(float)

   *(83)* size of central stop as a fraction of aperture.size

   If not None: places a central beam stop in aperture. The value given here is the fraction of the beam stop compared to `size` 

   *default* = ``None (>0.0, <1.0)``

.. py:data:: .scan.illumination.aperture.offset(float, tuple)

   *(84)* Offset between center of aperture and optical axes

   May also be a tuple (vertical,horizontal) for size in case of an asymmetric offset

   *default* = ``0``

.. py:data:: .scan.illumination.propagation(Param)

   *(85)* Parameters for propagation after aperture plane

   Propagation to focus takes precedence to parallel propagation if `foccused` is not ``None``

   *default* = ``None``

.. py:data:: .scan.illumination.propagation.parallel(float)

   *(86)* Parallel propagation distance

   If ``None`` or ``0`` : No parallel propagation 

   *default* = ``None``

.. py:data:: .scan.illumination.propagation.focussed(float)

   *(87)* Propagation distance from aperture to focus

   If ``None`` or ``0`` : No focus propagation 

   *default* = ``None``

.. py:data:: .scan.illumination.propagation.antialiasing(float)

   *(88)* Antialiasing factor

   Antialiasing factor used when generating the probe. (numbers larger than 2 or 3 are memory hungry)
   **[Untested]**

   *default* = ``1``

.. py:data:: .scan.illumination.propagation.spot_size(float)

   *(89)* Focal spot diameter

   If not ``None``, this parameter is used to generate the appropriate aperture size instead of :py:data:`size`

   *default* = ``None (>0.0)``

.. py:data:: .scan.illumination.diversity(Param)

   *(90)* Probe mode(s) diversity parameters

   Can be ``None`` i.e. no diversity

   *default* = ``None``

.. py:data:: .scan.illumination.diversity.noise(tuple)

   *(91)* Noise in the generated modes of the illumination 

   Can be either:
    - ``None`` : no noise
    - ``2-tuple`` : noise in phase (amplitude (rms), minimum feature size)
    - ``4-tuple`` : noise in phase & modulus (rms, mfs, rms_mod, mfs_mod)

   *default* = ``None``

.. py:data:: .scan.illumination.diversity.power(tuple, float)

   *(92)* Power of modes relative to main mode (zero-layer)

   

   *default* = ``0.1``

.. py:data:: .scan.illumination.diversity.shift(float)

   *(93)* Lateral shift of modes relative to main mode

   **[not implemented]**

   *default* = ``None``


.scan.sample
------------

.. py:data:: .scan.sample(Param)

   *(94)* Initial object modelization parameters

   In script, you may pass a numpy.array here directly as the model. This array will be passed to the storage instance with no checking whatsoever. Used in `~ptypy.core.sample`

   *default* = ``None (>0.0)``

.. py:data:: .scan.sample.model(str)

   *(95)* Type of initial object model

   One of:
    - ``None`` : model initialitziation defaults to flat array filled `fill`
    - ``'recon'`` : load model from STXM analysis of diffraction data
    - ``'stxm'`` : Estimate model from autocorrelation of mean diffraction data
    - *<resource>* : one of ptypys internal model resource strings
    - *<template>* : one of the templates in sample module
   
   In script, you may pass a numpy.array here directly as the model. This array will be processed according to `process` in order to *simulate* a sample from e.g. a thickness profile.

   *default* = ``None``

.. py:data:: .scan.sample.fill(float, complex)

   *(96)* Default fill value

   

   *default* = ``1``

.. py:data:: .scan.sample.recon(Param)

   *(97)* Parameters to load from previous reconstruction

   

   *default* = ``None``

.. py:data:: .scan.sample.recon.rfile(file)

   *(98)* Path to a ``.ptyr`` compatible file

   

   *default* = ``\*.ptyr``

.. py:data:: .scan.sample.recon.ID(NoneType)

   *(99)* ID (label) of storage data to load

   ``None`` is any ID

   *default* = ``None``

.. py:data:: .scan.sample.recon.layer(float)

   *(100)* Layer (mode) of storage data to load

   ``None`` is all layers, choose ``0`` for main mode

   *default* = ``None``

.. py:data:: .scan.sample.stxm(Param)

   *(101)* STXM analysis parameters

   

   *default* = ``None``

.. py:data:: .scan.sample.stxm.label(str)

   *(102)* Scan label of diffraction that is to be used for probe estimate

   ``None``, own scan label is used

   *default* = ``None``

.. py:data:: .scan.sample.process(Param)

   *(103)* Model processing parameters

   Can be ``None``, i.e. no processing

   *default* = ``None``

.. py:data:: .scan.sample.process.offset(tuple)

   *(104)* Offset between center of object array and scan pattern

   

   *default* = ``(0,0) (>0.0)``

.. py:data:: .scan.sample.process.zoom(tuple)

   *(105)* Zoom value for object simulation.

   If ``None``, leave the array untouched. Otherwise the modeled or loaded image will be resized using :py:func:`zoom`.

   *default* = ``None (>0.0)``

.. py:data:: .scan.sample.process.formula(str)

   *(106)* Chemical formula

   A Formula compatible with a cxro database query,e.g. ``'Au'`` or ``'NaCl'`` or ``'H2O'`` 

   *default* = ``None``

.. py:data:: .scan.sample.process.density(float)

   *(107)* Density in [g/ccm]

   Only used if `formula` is not None

   *default* = ``1``

.. py:data:: .scan.sample.process.thickness(float)

   *(108)* Maximum thickness of sample

   If ``None``, the absolute values of loaded source array will be used

   *default* = ``1.00E-06``

.. py:data:: .scan.sample.process.ref_index(complex)

   *(109)* Assigned refractive index

   If ``None``, treat source array as projection of refractive index. If a refractive index is provided the array's absolute value will be used to scale the refractive index.

   *default* = ``0.5+0.j (>0.0)``

.. py:data:: .scan.sample.process.smoothing(int)

   *(110)* Smoothing scale

   Smooth the projection with gaussian kernel of width given by `smoothing_mfs`

   *default* = ``2 (>0)``

.. py:data:: .scan.sample.diversity(Param)

   *(111)* Probe mode(s) diversity parameters

   Can be ``None`` i.e. no diversity

   *default* = ``None``

.. py:data:: .scan.sample.diversity.noise(tuple)

   *(112)* Noise in the generated modes of the illumination 

   Can be either:
    - ``None`` : no noise
    - ``2-tuple`` : noise in phase (amplitude (rms), minimum feature size)
    - ``4-tuple`` : noise in phase & modulus (rms, mfs, rms_mod, mfs_mod)

   *default* = ``None``

.. py:data:: .scan.sample.diversity.power(tuple, float)

   *(113)* Power of modes relative to main mode (zero-layer)

   

   *default* = ``0.1``

.. py:data:: .scan.sample.diversity.shift(float)

   *(114)* Lateral shift of modes relative to main mode

   **[not implemented]**

   *default* = ``None``


.scan.coherence
---------------

.. py:data:: .scan.coherence(Param)

   *(115)* Coherence parameters

   

   *default* = ``None (>0.0)``

.. py:data:: .scan.coherence.num_probe_modes(int)

   *(116)* Number of probe modes

   

   *default* = ``1 (>0)``

.. py:data:: .scan.coherence.num_object_modes(int)

   *(117)* Number of object modes

   

   *default* = ``1 (>0)``

.. py:data:: .scan.coherence.spectrum(list)

   *(118)* Amplitude of relative energy bins if the probe modes have a different energy

   

   *default* = ``None (>0.0)``

.. py:data:: .scan.coherence.object_dispersion(str)

   *(119)* Energy dispersive response of the object

   One of:
    - ``None`` or ``'achromatic'``: no dispersion
    - ``'linear'``: linear response model
    - ``'irregular'``: no assumption
   
   **[not implemented]**

   *default* = ``None``

.. py:data:: .scan.coherence.probe_dispersion(str)

   *(120)* Energy dispersive response of the probe

   One of:
    - ``None`` or ``'achromatic'``: no dispersion
    - ``'linear'``: linear response model
    - ``'irregular'``: no assumption
   
   **[not implemented]**

   *default* = ``None``


.scans
======

.. py:data:: .scans(Param)

   *(121)* Param container for instances of `scan` parameters

   If not specified otherwise, entries in *scans* will use parameter defaults from :py:data:`.scan`

   *default* = ``None``

.. py:data:: .scans.scan_00(scan)

   *(122)* Default first scans entry

   If only a single scan is used in the reconstruction, this entry may be left unchanged. If more than one scan is used, please make an entry for each scan. The name *scan_00* is an arbitrary choice and may be set to any other string.

   *default* = ``None``


.engine
=======

.. py:data:: .engine(Param)

   *(123)* Reconstruction engine parameters

   

   *default* = ``None``


.engine.common
--------------

.. py:data:: .engine.common(Param)

   *(124)* Parameters common to all engines

   

   *default* = ``None``

.. py:data:: .engine.common.name(str)

   *(125)* Name of engine. 

   Dependent on the name given here, the default parameter set will be a superset of `common` and parameters to the entry of the same name.

   *default* = ``DM``

.. py:data:: .engine.common.numiter(int)

   *(126)* Total number of iterations

   

   *default* = ``2000 (>0)``

.. py:data:: .engine.common.numiter_contiguous(int)

   *(127)* Number of iterations without interruption

   The engine will not return control to the caller until this number of iterations is completed (not processing server requests, I/O operations, ...)

   *default* = ``1 (>0)``

.. py:data:: .engine.common.probe_support(float)

   *(128)* Fraction of valid probe area (circular) in probe frame

   

   *default* = ``0.7 (>0.0)``

.. py:data:: .engine.common.clip_object(tuple)

   *(129)* Clip object amplitude into this intrervall

   

   *default* = ``None (>0.0)``


.engine.DM
----------

.. py:data:: .engine.DM(Param)

   *(130)* Parameters for Difference map engine

   

   *default* = ``None``

.. py:data:: .engine.DM.alpha(int)

   *(131)* Difference map parameter

   

   *default* = ``1 (>0)``

.. py:data:: .engine.DM.probe_update_start(int)

   *(132)* Number of iterations before probe update starts

   

   *default* = ``2 (>0)``

.. py:data:: .engine.DM.update_object_first(bool)

   *(133)* If False update object before probe

   

   *default* = ``TRUE (>0.0)``

.. py:data:: .engine.DM.overlap_converge_factor(float)

   *(134)* Threshold for interruption of the inner overlap loop

   The inner overlap loop refines the probe and the object simultaneously. This loop is escaped as soon as the overall change in probe, relative to the first iteration, is less than this value.

   *default* = ``0.05 (>0.0)``

.. py:data:: .engine.DM.overlap_max_iterations(int)

   *(135)* Maximum of iterations for the overlap constraint inner loop

   

   *default* = ``10 (>0)``

.. py:data:: .engine.DM.probe_inertia(float)

   *(136)* Weight of the current probe estimate in the update

   

   *default* = ``0.001 (>0.0)``

.. py:data:: .engine.DM.object_inertia(float)

   *(137)* Weight of the current object in the update

   

   *default* = ``0.1 (>0.0)``

.. py:data:: .engine.DM.fourier_relax_factor(float)

   *(138)* If rms error of model vs diffraction data is smaller than this fraction, Fourier constraint is met

   Set this value higher for noisy data

   *default* = ``0.01 (>0.0)``

.. py:data:: .engine.DM.obj_smooth_std(int)

   *(139)* Gaussian smoothing (pixel) of the current object prior to update

   If None, smoothing is deactivated. This smoothing can be used to reduce the amplitude of spurious pixels in the outer, least constrained areas of the object.

   *default* = ``20 (>0)``


.engine.ML
----------

.. py:data:: .engine.ML(Param)

   *(140)* Maximum Likelihood parameters

   

   *default* = ``None``

.. py:data:: .engine.ML.type(str)

   *(141)* Likelihood model. One of 'gaussian', 'poisson' or 'euclid'

   [only 'gaussian' is implemented for now]

   *default* = ``gaussian (>0.0)``

.. py:data:: .engine.ML.floating_intensities(bool)

   *(142)* If True, allow for adaptative rescaling of the diffraction pattern intensities (to correct for incident beam intensity fluctuations).

   

   *default* = ``FALSE``

.. py:data:: .engine.ML.intensity_renormalization(float)

   *(143)* A rescaling of the intensity so they can be interpreted as Poisson counts.

   

   *default* = ``1 (>0.0)``

.. py:data:: .engine.ML.reg_del2(bool)

   *(144)* Whether to use a Gaussian prior (smoothing) regularizer.

   

   *default* = ``TRUE (>0.0)``

.. py:data:: .engine.ML.reg_del2_amplitude(float)

   *(145)* Amplitude of the Gaussian prior if used.

   

   *default* = ``0.01 (>0.0)``

.. py:data:: .engine.ML.smooth_gradient(float)

   *(146)* Smoothing preconditioner. If 0, not used, if > 0 gaussian filter if < 0 Hann window.

   

   *default* = ``0 (>0.0)``

.. py:data:: .engine.ML.scale_precond(bool)

   *(147)* Whether to use the object/probe scaling preconditioner.

   This parameter can give faster convergence for weakly scattering samples.

   *default* = ``FALSE (>0.0)``

.. py:data:: .engine.ML.scale_probe_object(float)

   *(148)* Relative scale of probe to object.

   

   *default* = ``1 (>0.0)``

.. py:data:: .engine.ML.probe_update_start(int)

   *(149)* Number of iterations before probe update starts

   

   *default* = ``0``


.engines
========

.. py:data:: .engines(Param)

   *(150)* Container for instances of "engine" parameters

   All engines registered in this structure will be executed sequentially.

   *default* = ``None``

.. py:data:: .engines.engine_00(engine)

   *(151)* Default first engines entry

   Default first engine is difference map (DM)

   *default* = ``None``

