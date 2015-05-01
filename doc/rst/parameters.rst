Ptypy parameter structure
=========================

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
---

.. py:data:: .io(Param)

   *(4)* Global parameters for I/O

   

   *default* = ``None``

.. py:data:: .io.paths(Param)

   *(5)* Paths

   The paths parameters can contain format strings with keywords from the runtime directory. The most likely are "run", "engine", "iterations".

   *default* = ``None``

.. py:data:: .io.paths.home(dir)

   *(6)* Base directory for all I/O

   home is the root directory for all input/output operations. All other path parameters that are relative paths will be relative to this directory.

   *default* = ``./``

.. py:data:: .io.paths.recons(str)

   *(7)* Reconstruction file name (or format string)

   Reconstruction file name or format string (constructed against runtime dictionary)

   *default* = ``recons/%(run)s/%(run)s_%(engine)s_%(iterations)04d.ptyr``

.. py:data:: .io.paths.autosave(str)

   *(8)* Auto-save file name (or format string)

   Auto-save file name or format string (constructed against runtime dictionary)

   *default* = ``dumps/%(run)s/%(run)s_%(engine)s_%(iterations)04d.ptyr``

.. py:data:: .io.paths.data(str)

   *(9)* Processed data file name (or format string)

   

   *default* = ``analysis/%(run)s/%...``

.. py:data:: .io.paths.plots(str)

   *(10)* Plot images file name (or format string)

   

   *default* = ``plots/%(run)s/%(run)s_%(engine)s_%(iterations)04d.png``

.. py:data:: .io.paths.movie(str)

   *(11)* Movie file name (or format string)

   

   *default* = ``plots/%(run)s/%(run)s_%(engine)s.mpg``

.. py:data:: .io.autosave(Param)

   *(12)* Auto-save options

   

   *default* = ``None``

.. py:data:: .io.autosave.active(bool)

   *(13)* Activation switch

   If ``True`` the current reconstruction will be saved at regular intervals according to the pattern in :py:data:`paths.autosave`

   *default* = ``TRUE``

.. py:data:: .io.autosave.interval(int)

   *(14)* Auto-save interval

   

   *default* = ``10 (>1)``


.interaction
------------

.. py:data:: .interaction(Param)

   *(15)* Server / Client parameters

   If ``None`` is passed here in script instead of a Param, it translates to  ``active=False`` i.e. no ZeroMQ interaction server. 

   *default* = ``None``

.. py:data:: .interaction.active(bool)

   *(16)* Activation switch

   Set to ``False`` for no  ZeroMQ interaction server
   

   *default* = ``TRUE``

.. py:data:: .interaction.primary_address(str)

   *(17)* The address the server is listening to.

   The address the server is listening to.
   TODO: allow for automatic address definition when running on a cluster.

   *default* = ``tcp://127.0.0.2``

.. py:data:: .interaction.primary_port(int)

   *(18)* The port the server is listening to.

   The port the server is listening to.

   *default* = ``5570``

.. py:data:: .interaction.port_range(str)

   *(19)* The port range opened to clients.

   The port range opened to clients.

   *default* = ``5664:00:00``


.plotclient
-----------

.. py:data:: .plotclient(Param)

   *(20)* Plotting client parameters

   

   *default* = ``None``

.. py:data:: .plotclient.active(bool)

   *(21)* Live plotting switch

   If True, a plotting client will be spawned and connected at initialization. This option should be set to False when ptypy is run on a cluster.

   *default* = ``TRUE``

.. py:data:: .plotclient.interval(int)

   *(22)* Number of iterations between plot updates

   Requests to the server will happen with this iteration intervals. Note that this will work only if interaction.polling_interval is smaller or equal to this number.

   *default* = ``1 (>1)``

.. py:data:: .plotclient.some_plotting_options(str)

   *(23)* Options for default plotter (not implemented yet)

   Options for default plotter (not implemented yet)

   *default* = ``None``

.. py:data:: .plotclient.dump(bool)

   *(24)* Switch to dump plots as image files

   

   *default* = ``TRUE``

.. py:data:: .plotclient.dump_interval(int)

   *(25)* Iteration interval for dumping plots

   If None, no image will be saved. If 0, only a final image will be saved.

   *default* = ``None``

.. py:data:: .plotclient.make_movie(bool)

   *(26)* Produce reconstruction movie after the reconstruction.

   Switch to request the production of a movie from the dumped plots at the end of the reconstruction.

   *default* = ``TRUE``


.scan
-----

.. py:data:: .scan(Param)

   *(27)* Scan parameters

   This categrogy specifies defaults for all scans. Scan-specific parameters are stored in scans.scan_%%

   *default* = ``None``

.. py:data:: .scan.tags(str)

   *(28)* Comma seperated string tags describing the data input

   [deprecated?]

   *default* = ``None``

.. py:data:: .scan.if_conflict_use_meta(bool)

   *(29)* Give priority to metadata relative to input parameters

   [useful?]

   *default* = ``TRUE``

.. py:data:: .scan.data(Param)

   *(30)* Data preparation parameters

   

   *default* = ``None``

.. py:data:: .scan.data.recipe(ext)

   *(31)* Data preparation recipe container

   

   *default* = ``None``

.. py:data:: .scan.data.source(file)

   *(32)* Describes where to get the data from.


   Accepted values are:
    - ``'file'``: data will be read from a .ptyd file.
    - any valid recipe name: data will be prepared using the recipe.
    - ``'sim'`` : data will be simulated according to parameters in simulation  

   *default* = ``None``

.. py:data:: .scan.data.dfile(file)

   *(33)* Prepared data file path

   If source was ``None`` or ``'file'``, data will be loaded from this file and processing as well as saving is deactivated. If source is the name of an experiment recipe or path to a file, data will be saved to this file

   *default* = ``None``

.. py:data:: .scan.data.label(str)

   *(34)* The scan label

   Unique string identifying the scan

   *default* = ``None``

.. py:data:: .scan.data.shape(int, tuple)

   *(35)* Shape of the region of interest cropped from the raw data.

   Cropping dimension of the diffraction frame
   Can be None, (dimx, dimy), or dim. In the latter case shape will be (dim, dim).

   *default* = ``None``

.. py:data:: .scan.data.save(str)

   *(36)* Saving mode

   Mode to use to save data to file.
    - ``None``: No saving 
    - ``'merge'``: attemts to merge data in single chunk **[not implemented]**
    - ``'append'``: appends each chunk in master *.ptyd file
    - ``'link'``: appends external links in master *.ptyd file and stores chunks separately in the path given by the link. Links file paths are relative to master file.

   *default* = ``None``

.. py:data:: .scan.data.center(tuple)

   *(37)* Center (pixel) of the optical axes in raw data

   If ``None``, this parameter will be set by :py:data:`~.scan.data.auto_center` or elsewhere

   *default* = ``None``

.. py:data:: .scan.data.psize(float, tuple)

   *(38)* Detector pixel size

   Dimensions of the detector pixels (in meters)

   *default* = ``None (>0.0)``

.. py:data:: .scan.data.distance(float)

   *(39)* Sample-to-detector distance

   In meters.

   *default* = ``None (>0.0)``

.. py:data:: .scan.data.rebin(int)

   *(40)* Rebinning factor

   Rebinning factor for the raw data frames. ``'None'`` or ``1`` both mean *no binning*

   *default* = ``None (>1, <8)``

.. py:data:: .scan.data.orientation(int, tuple)

   *(41)* Data frame orientation

    - ``None`` or ``0``: correct orientation
    - ``1``: invert columns (numpy.flip_lr)
    - ``2``: invert columns, invert rows
    - ``3``: invert rows  (numpy.flip_ud)
    - ``4``: transpose (numpy.transpose)
    - ``4+i``: tranpose + other operations from above
   
   Alternatively, a 3-tuple of booleans may be provided ``(do_transpose, do_flipud, do_fliplr)``

   *default* = ``None``

.. py:data:: .scan.data.energy(float)

   *(42)* Photon energy of the incident radiation

   

   *default* = ``None (>0.0)``

.. py:data:: .scan.data.min_frames(int)

   *(43)* Minimum number of frames loaded by each node

   

   *default* = ``1``

.. py:data:: .scan.data.num_frames(int)

   *(44)* Maximum number of frames to be prepared

   If `positions_theory` are provided, num_frames will be ovverriden with the number of positions available

   *default* = ``None``

.. py:data:: .scan.data.chunk_format(str)

   *(45)* Appendix to saved files if save == 'link'

   

   *default* = ``.chunk%02d``

.. py:data:: .scan.data.auto_center(bool)

   *(46)* Determine if center in data is calculated automatically

    - ``False``, no automatic centering 
    - ``None``, only if :any:`center` is ``None`` 
    - ``True``, it will be enforced

   *default* = ``None``

.. py:data:: .scan.data.load_parallel(str)

   *(47)* Determines what will be loaded in parallel

   Choose from ``None``, ``'data'``, ``'common'``, ``'all'``

   *default* = ``data``

.. py:data:: .scan.data.positions_theory(ndarray)

   *(48)* Theoretical positions for this scan

   If provided, experimental positions from :any:`Ptyscan` subclass will be ignored. If data preparation is called from Ptycho instance, the calculated positions from the :py:func:`ptypy.core.xy.from_pars` dict will be inserted here

   *default* = ``None``

.. py:data:: .scan.data.experimentID(str)

   *(49)* Name of the experiment

   If None, a default value will be provided by the recipe.

   *default* = ``None``

.. py:data:: .scan.data.simulation(Param)

   *(50)* Simulated data as a preparation

   Similar to scan, simulation takes Parameters trees in the same form `illumination`, `sample` and `xy`. Any item in these trees will take precedence over scan specific parameters in the simulated scan.

   *default* = ``None``

.. py:data:: .scan.data.simulation.detector(Param, str, NoneType)

   *(51)* Detector parameters

   Can also be ``None`` if no detector specific filter is wanted or a string that matches one of the templates in the detector module

   *default* = ``None``

.. py:data:: .scan.data.simulation.psf(float)

   *(52)* Gaussian point spread in detector

   Value passed here represents the FWHM of a Gaussian. ``None`` means no point spread.

   *default* = ``None``

.. py:data:: .scan.sharing(Param)

   *(53)* Scan sharing options

   

   *default* = ``None``

.. py:data:: .scan.sharing.object_share_with(str)

   *(54)* Label or index of scan to share object with.

   Possible values:
    - ``None``: Do not share
    - *(string)*: Label of the scan to share with
    - *(int)*: Index of scan to share with

   *default* = ``None``

.. py:data:: .scan.sharing.object_share_power(float)

   *(55)* Relative power for object sharing

   

   *default* = ``1 (>0.0)``

.. py:data:: .scan.sharing.probe_share_with(str)

   *(56)* Label or index of scan to share probe with.

   Possible values:
    - ``None``: Do not share
    - *(string)*: Label of the scan to share with
    - *(int)*: Index of scan to share with

   *default* = ``None``

.. py:data:: .scan.sharing.probe_share_power(float)

   *(57)* Relative power for probe sharing

   

   *default* = ``1 (>0.0)``

.. py:data:: .scan.geometry(Param)

   *(58)* Physical parameters

   All distances are in meters. Other units are specified in the documentation strings.

   *default* = ``None``

.. py:data:: .scan.geometry.energy(float)

   *(59)* Energy (in keV)

   If ``None``, uses `lam` instead.

   *default* = ``6.2 (>0.0)``

.. py:data:: .scan.geometry.lam(float)

   *(60)* Wavelength

   Used only if `energy` is ``None``

   *default* = ``None (>0.0)``

.. py:data:: .scan.geometry.distance(float)

   *(61)* Distance from object to detector

   

   *default* = ``7.19 (>0.0)``

.. py:data:: .scan.geometry.psize(float)

   *(62)* Pixel size in Detector plane

   

   *default* = ``0.000172 (>0.0)``

.. py:data:: .scan.geometry.resolution(float)

   *(63)* Pixel size in Sample plane

   This parameter is used only for simulations

   *default* = ``None (>0.0)``

.. py:data:: .scan.geometry.prop_type(str)

   *(64)* Propagation type

   Either "farfield" or "nearfield"

   *default* = ``farfield``

.. py:data:: .scan.xy(Param)

   *(65)* Parameters for scan patterns

   These parameters are useful in two cases:
    - When the experimental positions are not known (no encoders)
    - When using the package to simulate data.
   
   In script an array of shape *(N,2)* may be passed here instead of a Param or dictionary as an **override**

   *default* = ``None``

.. py:data:: .scan.xy.model(str)

   *(66)* Scan pattern type

   The type must be one of the following:
    - ``None``: positions are read from data file.
    - ``'raster'``: raster grid pattern
    - ``'round'``: concentric circles pattern
    - ``'spiral'``: spiral pattern
   
   In script an array of shape *(N,2)* may be passed here instead

   *default* = ``None (>0.0)``

.. py:data:: .scan.xy.spacing(float, tuple)

   *(67)* Pattern spacing

   Spacing between scan positions. If the model supports asymmetric scans, a tuple passed here will be interpreted as *(dy,dx)* with *dx* as horizontal spacing and *dy* as vertical spacing. If ``None`` the value is calculated from `extent` and `steps`
   

   *default* = ``1.50E-06 (>0.0)``

.. py:data:: .scan.xy.steps(int, tuple)

   *(68)* Pattern step count

   Number of steps with length *spacing* in the grid. A tuple *(ny,nx)* provided here can be used for a different step in vertical ( *ny* ) and horizontal direction ( *nx* ). If ``None`` the, step count is calculated from `extent` and `spacing`

   *default* = ``10 (>0)``

.. py:data:: .scan.xy.extent(float, tuple)

   *(69)* Rectangular extent of pattern

   Defines the absolut maximum extent. If a tuple *(ly,lx)* is provided the extent may be rectangular rather than square. All positions outside of `extent` will be discarded. If ``None`` the extent will is `spacing` times `steps`

   *default* = ``1.50E-05 (>0.0)``

.. py:data:: .scan.xy.offset(float, tuple)

   *(70)* Offset of scan pattern relative to origin


   If tuple, the offset may differ in *x* and *y*. Please not that the offset will be included when removing scan points outside of `extend`.

   *default* = ``0``

.. py:data:: .scan.xy.jitter(float, tuple)

   *(71)* RMS of jitter on sample position

   **Only use in simulation**. Adds a random jitter to positions.

   *default* = ``0``

.. py:data:: .scan.xy.count(int)

   *(72)* Number of scan points


   Only return return positions up to number of `count`.

   *default* = ``None``

.. py:data:: .scan.illumination(Param)

   *(73)* Illumination model (probe)

   
   In script, you may pass directly a three dimensional  numpy.ndarray here instead of a `Param`. This array will be copied to the storage instance with no checking whatsoever. Used in `~ptypy.core.illumination`

   *default* = ``None (>0.0)``

.. py:data:: .scan.illumination.model(str)

   *(74)* Type of illumination model

   One of:
    - ``None`` : model initialitziation defaults to flat array filled with the specified number of photons
    - ``'recon'`` : load model from previous reconstruction, see `recon` Parameters
    - ``'stxm'`` : Estimate model from autocorrelation of mean diffraction data
    - *<resource>* : one of ptypys internal image resource strings
    - *<template>* : one of the templates inillumination module
   
   In script, you may pass a numpy.ndarray here directly as the model. It is considered as incoming wavefront and will be propagated according to `propagation` with an optional `aperture` applied before

   *default* = ``None``

.. py:data:: .scan.illumination.photons(int)

   *(75)* Number of photons in the incident illumination

   A value specified here will take precedence over calculated statistics from the loaded data.

   *default* = ``None (>0)``

.. py:data:: .scan.illumination.recon(Param)

   *(76)* Parameters to load from previous reconstruction

   

   *default* = ``None``

.. py:data:: .scan.illumination.recon.rfile(file)

   *(77)* Path to a ``.ptyr`` compatible file

   

   *default* = ``*.ptyr``

.. py:data:: .scan.illumination.recon.ID(NoneType)

   *(78)* ID (label) of storage data to load

   ``None`` means any ID

   *default* = ``None``

.. py:data:: .scan.illumination.recon.layer(float)

   *(79)* Layer (mode) of storage data to load

   ``None`` means all layers, choose ``0`` for main mode

   *default* = ``None``

.. py:data:: .scan.illumination.stxm(Param)

   *(80)* Parameters to initialize illumination from diffraction data

   

   *default* = ``None``

.. py:data:: .scan.illumination.stxm.label(str)

   *(81)* Scan label of diffraction that is to be used for probe estimate

   ``None``, own scan label is used

   *default* = ``None``

.. py:data:: .scan.illumination.aperture(Param)

   *(82)* Beam aperture parameters

   

   *default* = ``None``

.. py:data:: .scan.illumination.aperture.form(str)

   *(83)* One of None, 'rect' or 'circ'

   One of:
    - ``None`` : no aperture, this may be useful for nearfield
    - ``'rect'`` : rectangular aperture
    - ``'circ'`` : circular aperture

   *default* = ``circ (>0.0)``

.. py:data:: .scan.illumination.aperture.diffuser(float)

   *(84)* Noise in the transparen part of the aperture

   Can be either:
   - ``None`` : no noise
   - ``2-tuple`` : noise in phase (amplitude (rms), minimum feature size)
   - ``4-tuple`` : noise in phase & modulus (rms, mfs, rms_mod, mfs_mod)

   *default* = ``None (>0.0)``

.. py:data:: .scan.illumination.aperture.size(float)

   *(85)* Aperture width or diameter

   May also be a tuple *(vertical,horizontal)* in case of an asymmetric aperture 

   *default* = ``None (>0.0)``

.. py:data:: .scan.illumination.aperture.edge(int)

   *(86)* Edge width of aperture (in pixels!)

   

   *default* = ``2 (>0)``

.. py:data:: .scan.illumination.aperture.central_stop(float)

   *(87)* size of central stop as a fraction of aperture.size

   If not None: places a central beam stop in aperture. The value given here is the fraction of the beam stop compared to `size` 

   *default* = ``None (>0.0, <1.0)``

.. py:data:: .scan.illumination.aperture.offset(float, tuple)

   *(88)* Offset between center of aperture and optical axes

   May also be a tuple (vertical,horizontal) for size in case of an asymmetric offset

   *default* = ``0``

.. py:data:: .scan.illumination.propagation(Param)

   *(89)* Parameters for propagation after aperture plane

   Propagation to focus takes precedence to parallel propagation if `foccused` is not ``None``

   *default* = ``None``

.. py:data:: .scan.illumination.propagation.parallel(float)

   *(90)* Parallel propagation distance

   If ``None`` or ``0`` : No parallel propagation 

   *default* = ``None``

.. py:data:: .scan.illumination.propagation.focussed(float)

   *(91)* Propagation distance from aperture to focus

   If ``None`` or ``0`` : No focus propagation 

   *default* = ``None``

.. py:data:: .scan.illumination.propagation.antialiasing(float)

   *(92)* Antialiasing factor

   Antialiasing factor used when generating the probe. (numbers larger than 2 or 3 are memory hungry)
   **[Untested]**

   *default* = ``1``

.. py:data:: .scan.illumination.propagation.spot_size(float)

   *(93)* Focal spot diameter

   If not ``None``, this parameter is used to generate the appropriate aperture size instead of :any:`size`

   *default* = ``None (>0.0)``

.. py:data:: .scan.illumination.diversity(Param)

   *(94)* Probe mode(s) diversity parameters

   Can be ``None`` i.e. no diversity

   *default* = ``None``

.. py:data:: .scan.illumination.diversity.noise(tuple)

   *(95)* Noise in the generated modes of the illumination 

   Can be either:
   - ``None`` : no noise
   - ``2-tuple`` : noise in phase (amplitude (rms), minimum feature size)
   - ``4-tuple`` : noise in phase & modulus (rms, mfs, rms_mod, mfs_mod)

   *default* = ``None``

.. py:data:: .scan.illumination.diversity.power(tuple, float)

   *(96)* Power of modes relative to main mode (zero-layer)

   

   *default* = ``0.1``

.. py:data:: .scan.illumination.diversity.shift(float)

   *(97)* Lateral shift of modes relative to main mode

   **[not implemented]**

   *default* = ``None``

.. py:data:: .scan.sample(Param)

   *(98)* Initial object modelization parameters

   In script, you may pass a numpy.array here directly as the model. This array will be passed to the storage instance with no checking whatsoever. Used in `~ptypy.core.sample`

   *default* = ``None (>0.0)``

.. py:data:: .scan.sample.model(str)

   *(99)* Type of initial object model

   One of:
    - ``None`` : model initialitziation defaults to flat array filled `fill`
    - ``'recon'`` : load model from STXM analysis of diffraction data
    - ``'stxm'`` : Estimate model from autocorrelation of mean diffraction data
    - *<resource>* : one of ptypys internal model resource strings
    - *<template>* : one of the templates in sample module
   
   In script, you may pass a numpy.array here directly as the model. This array will be processed according to `process` in order to *simulate* a sample from e.g. a thickness profile.

   *default* = ``None``

.. py:data:: .scan.sample.fill(float, complex)

   *(100)* Default fill value

   

   *default* = ``1``

.. py:data:: .scan.sample.recon(Param)

   *(101)* Parameters to load from previous reconstruction

   

   *default* = ``None``

.. py:data:: .scan.sample.recon.rfile(file)

   *(102)* Path to a ``.ptyr`` compatible file

   

   *default* = ``*.ptyr``

.. py:data:: .scan.sample.recon.ID(NoneType)

   *(103)* ID (label) of storage data to load

   ``None`` is any ID

   *default* = ``None``

.. py:data:: .scan.sample.recon.layer(float)

   *(104)* Layer (mode) of storage data to load

   ``None`` is all layers, choose ``0`` for main mode

   *default* = ``None``

.. py:data:: .scan.sample.stxm(Param)

   *(105)* STXM analysis parameters

   

   *default* = ``None``

.. py:data:: .scan.sample.stxm.label(str)

   *(106)* Scan label of diffraction that is to be used for probe estimate

   ``None``, own scan label is used

   *default* = ``None``

.. py:data:: .scan.sample.process(Param)

   *(107)* Model processing parameters

   Can be ``None``, i.e. no processing

   *default* = ``None``

.. py:data:: .scan.sample.process.offset(tuple)

   *(108)* Offset between center of object array and scan pattern

   

   *default* = ``(0,0) (>0.0)``

.. py:data:: .scan.sample.process.zoom(tuple)

   *(109)* Zoom value for object simulation.

   If ``None``, leave the array untouched. Otherwise the modeled or loaded image will be resized using :any:`zoom`.

   *default* = ``None (>0.0)``

.. py:data:: .scan.sample.process.formula(str)

   *(110)* Chemical formula

   A Formula compatible with a cxro database query,e.g. ``'Au'`` or ``'NaCl'`` or ``'H2O'`` 

   *default* = ``None``

.. py:data:: .scan.sample.process.density(float)

   *(111)* Density in [g/ccm]

   Only used if `formula` is not None

   *default* = ``1``

.. py:data:: .scan.sample.process.thickness(float)

   *(112)* Maximum thickness of sample

   If ``None``, the absolute values of loaded source array will be used

   *default* = ``1.00E-06``

.. py:data:: .scan.sample.process.ref_index(complex)

   *(113)* Assigned refractive index

   If ``None``, treat source array as projection of refractive index. If a refractive index is provided the array's absolute value will be used to scale the refractive index.

   *default* = ``0.5+0.j (>0.0)``

.. py:data:: .scan.sample.process.smoothing(int)

   *(114)* Smoothing scale

   Smooth the projection with gaussian kernel of width given by `smoothing_mfs`

   *default* = ``2 (>0)``

.. py:data:: .scan.sample.diversity(Param)

   *(115)* Probe mode(s) diversity parameters

   Can be ``None`` i.e. no diversity

   *default* = ``None``

.. py:data:: .scan.sample.diversity.noise(tuple)

   *(116)* Noise in the generated modes of the illumination 

   Can be either:
   - ``None`` : no noise
   - ``2-tuple`` : noise in phase (amplitude (rms), minimum feature size)
   - ``4-tuple`` : noise in phase & modulus (rms, mfs, rms_mod, mfs_mod)

   *default* = ``None``

.. py:data:: .scan.sample.diversity.power(tuple, float)

   *(117)* Power of modes relative to main mode (zero-layer)

   

   *default* = ``0.1``

.. py:data:: .scan.sample.diversity.shift(float)

   *(118)* Lateral shift of modes relative to main mode

   **[not implemented]**

   *default* = ``None``

.. py:data:: .scan.coherence(Param)

   *(119)* Coherence parameters

   

   *default* = ``None (>0.0)``

.. py:data:: .scan.coherence.num_probe_modes(int)

   *(120)* Number of probe modes

   

   *default* = ``1 (>0)``

.. py:data:: .scan.coherence.num_object_modes(int)

   *(121)* Number of object modes

   

   *default* = ``1 (>0)``

.. py:data:: .scan.coherence.spectrum(list)

   *(122)* Amplitude of relative energy bins if the probe modes have a different energy

   

   *default* = ``None (>0.0)``

.. py:data:: .scan.coherence.object_dispersion(str)

   *(123)* Energy dispersive response of the object

   One of:
    - ``None`` or ``'achromatic'``: no dispersion
    - ``'linear'``: linear response model
    - ``'irregular'``: no assumption
   
   **[not implemented]**

   *default* = ``None``

.. py:data:: .scan.coherence.probe_dispersion(str)

   *(124)* Energy dispersive response of the probe

   One of:
    - ``None`` or ``'achromatic'``: no dispersion
    - ``'linear'``: linear response model
    - ``'irregular'``: no assumption
   
   **[not implemented]**

   *default* = ``None``


.scans
------

.. py:data:: .scans(Param)

   *(125)* Param container for instances of `scan` parameters

   

   *default* = ``None``

.. py:data:: .scans.scan_%d(scan)

   *(126)* 

   

   *default* = ``None``


.engine
-------

.. py:data:: .engine(Param)

   *(127)* Reconstruction engine parameters

   

   *default* = ``None``

.. py:data:: .engine.common(Param)

   *(128)* Parameters common to all engines

   

   *default* = ``None``

.. py:data:: .engine.common.numiter(int)

   *(129)* Total number of iterations

   

   *default* = ``2000 (>0)``

.. py:data:: .engine.common.numiter_contiguous(int)

   *(130)* Number of iterations without interruption

   The engine will not return control to the caller until this number of iterations is completed (not processing server requests, I/O operations, ...)

   *default* = ``1 (>0)``

.. py:data:: .engine.common.probe_support(float)

   *(131)* Fraction of valid probe area (circular) in probe frame

   

   *default* = ``0.7 (>0.0)``

.. py:data:: .engine.common.clip_object(tuple)

   *(132)* Clip object amplitude into this intrervall

   

   *default* = ``None (>0.0)``

.. py:data:: .engine.DM(Param)

   *(133)* Parameters for Difference map engine

   

   *default* = ``None``

.. py:data:: .engine.DM.alpha(int)

   *(134)* Difference map parameter

   

   *default* = ``1 (>0)``

.. py:data:: .engine.DM.probe_update_start(int)

   *(135)* Number of iterations before probe update starts

   

   *default* = ``2 (>0)``

.. py:data:: .engine.DM.update_object_first(bool)

   *(136)* If False update object before probe

   

   *default* = ``TRUE (>0.0)``

.. py:data:: .engine.DM.overlap_converge_factor(float)

   *(137)* Threshold for interruption of the inner overlap loop

   The inner overlap loop refines the probe and the object simultaneously. This loop is escaped as soon as the overall change in probe, relative to the first iteration, is less than this value.

   *default* = ``0.05 (>0.0)``

.. py:data:: .engine.DM.overlap_max_iterations(int)

   *(138)* Maximum of iterations for the overlap constraint inner loop

   

   *default* = ``10 (>0)``

.. py:data:: .engine.DM.probe_inertia(float)

   *(139)* Weight of the current probe estimate in the update

   

   *default* = ``0.001 (>0.0)``

.. py:data:: .engine.DM.object_inertia(float)

   *(140)* Weight of the current object in the update

   

   *default* = ``0.1 (>0.0)``

.. py:data:: .engine.DM.fourier_relax_factor(float)

   *(141)* If rms error of model vs diffraction data is smaller than this fraction, Fourier constraint is met

   Set this value higher for noisy data

   *default* = ``0.01 (>0.0)``

.. py:data:: .engine.DM.obj_smooth_std(int)

   *(142)* Gaussian smoothing (pixel) of the current object prior to update

   If None, smoothing is deactivated. This smoothing can be used to reduce the amplitude of spurious pixels in the outer, least constrained areas of the object.

   *default* = ``20 (>0)``

.. py:data:: .engine.ML(Param)

   *(143)* Maximum Likelihood parameters

   

   *default* = ``None``

.. py:data:: .engine.ML.type(str)

   *(144)* Likelihood model. One of 'gaussian', 'poisson' or 'euclid'

   [only 'gaussian' is implemented for now]

   *default* = ``gaussian (>0.0)``

.. py:data:: .engine.ML.floating_intensities(bool)

   *(145)* If True, allow for adaptative rescaling of the diffraction pattern intensities (to correct for incident beam intensity fluctuations).

   

   *default* = ``FALSE``

.. py:data:: .engine.ML.intensity_renormalization(float)

   *(146)* A rescaling of the intensity so they can be interpreted as Poisson counts.

   

   *default* = ``1 (>0.0)``

.. py:data:: .engine.ML.reg_del2(bool)

   *(147)* Whether to use a Gaussian prior (smoothing) regularizer.

   

   *default* = ``TRUE (>0.0)``

.. py:data:: .engine.ML.reg_del2_amplitude(float)

   *(148)* Amplitude of the Gaussian prior if used.

   

   *default* = ``0.01 (>0.0)``

.. py:data:: .engine.ML.smooth_gradient(float)

   *(149)* Smoothing preconditioner. If 0, not used, if > 0 gaussian filter if < 0 Hann window.

   

   *default* = ``0 (>0.0)``

.. py:data:: .engine.ML.scale_precond(bool)

   *(150)* Whether to use the object/probe scaling preconditioner.

   This parameter can give faster convergence for weakly scattering samples.

   *default* = ``FALSE (>0.0)``

.. py:data:: .engine.ML.scale_probe_object(float)

   *(151)* Relative scale of probe to object.

   

   *default* = ``1 (>0.0)``

.. py:data:: .engine.ML.probe_update_start(int)

   *(152)* Number of iterations before probe update starts

   

   *default* = ``0``


.engines
--------

.. py:data:: .engines(Param)

   *(153)* Container for instances of "engine" parameters

   All engines registered in this structure will be executed sequentially.

   *default* = ``None``

.. py:data:: .engines.engine_%d(engine)

   *(154)* 

   

   *default* = ``None``

