Ptypy parameter structure
=========================

.. py:data:: .verbose_level(int)

   *(0)* Verbosity level

   Verbosity level for information logging
   
   0: Only errors
   1: Warning
   2: Information
   3: Debug

   *default* = ``1 (>0, <3)``

.. py:data:: .data_type(str)

   *(1)* Reconstruction floating number precision

   Reconstruction floating number precision ('single' or 'double')

   *default* = ``single``

.. py:data:: .run(str)

   *(2)* Reconstruction identifier

   Reconstruction run identifier. If None (default), the run name will be constructed at run time from other information.

   *default* = ``None``

.. py:data:: .dry_run(bool)

   *(3)* Dry run switch - NOT IMPLEMENTED

   Run everything skipping all memory and cpu-heavy steps (and file saving)

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

   If True the current reconstruction will be saved at regular intervals.

   *default* = ``TRUE``

.. py:data:: .io.autosave.interval(int)

   *(14)* Auto-save interval

   

   *default* = ``10 (>1)``


.interaction
------------

.. py:data:: .interaction(Param, NoneType)

   *(15)* Server / Client parameters

   Can be `None` i.e. no ZeroMQ interaction server

   *default* = ``None``

.. py:data:: .interaction.primary_address(str)

   *(16)* The address the server is listening to.

   The address the server is listening to.
   TODO: allow for automatic address definition when running on a cluster.

   *default* = ``tcp://127.0.0.2``

.. py:data:: .interaction.primary_port(int)

   *(17)* The port the server is listening to.

   The port the server is listening to.

   *default* = ``5570``

.. py:data:: .interaction.port_range(str)

   *(18)* The port range opened to clients.

   The port range opened to clients.

   *default* = ``5664:00:00``


.plotclient
-----------

.. py:data:: .plotclient(Param)

   *(19)* Plotting client parameters

   

   *default* = ``None``

.. py:data:: .plotclient.active(bool)

   *(20)* Live plotting switch

   If True, a plotting client will be spawned and connected at initialization. This option should be set to False when ptypy is run on a cluster.

   *default* = ``TRUE``

.. py:data:: .plotclient.interval(int)

   *(21)* Number of iterations between plot updates

   Requests to the server will happen with this iteration intervals. Note that this will work only if interaction.polling_interval is smaller or equal to this number.

   *default* = ``1 (>1)``

.. py:data:: .plotclient.some_plotting_options(str)

   *(22)* Options for default plotter (not implemented yet)

   Options for default plotter (not implemented yet)

   *default* = ``None``

.. py:data:: .plotclient.dump(bool)

   *(23)* Switch to dump plots as image files

   

   *default* = ``TRUE``

.. py:data:: .plotclient.dump_interval(int)

   *(24)* Iteration interval for dumping plots

   If None, no image will be saved. If 0, only a final image will be saved.

   *default* = ``None``

.. py:data:: .plotclient.make_movie(bool)

   *(25)* Produce reconstruction movie after the reconstruction.

   Switch to request the production of a movie from the dumped plots at the end of the reconstruction.

   *default* = ``TRUE``


.scan
-----

.. py:data:: .scan(Param)

   *(26)* Scan parameters

   This categrogy specifies defaults for all scans. Scan-specific parameters are stored in scans.scan_%%

   *default* = ``None``

.. py:data:: .scan.tags(str)

   *(27)* Comma seperated string tags describing the data input

   [deprecated?]

   *default* = ``None``

.. py:data:: .scan.if_conflict_use_meta(bool)

   *(28)* Give priority to metadata relative to input parameters

   [useful?]

   *default* = ``TRUE``

.. py:data:: .scan.data(Param)

   *(29)* Data preparation parameters

   

   *default* = ``None``

.. py:data:: .scan.data.recipe(ext)

   *(30)* Data preparation recipe container

   

   *default* = ``None``

.. py:data:: .scan.data.source(file)

   *(31)* Origin of data

   Describes where to get the data from.
   
   Accepted values are:
    - `file`: data will be read from a .ptyd file.
    - any valid recipe name: data will be prepared using the recipe.
    - `sim` : data will be simulated according to parameters in simulation  

   *default* = ``None``

.. py:data:: .scan.data.dfile(file)

   *(32)* Prepared data file path

   If source was None or `file`, data will be loaded from this file and processing as well as saving is deactivated. If source is the name of an experiment recipe or path to a file, data will be saved to this file

   *default* = ``None``

.. py:data:: .scan.data.label(str)

   *(33)* The scan label

   Unique string identifying the scan

   *default* = ``None``

.. py:data:: .scan.data.shape(int, tuple)

   *(34)* Shape of the region of interest cropped from the raw data.

   Cropping dimension of the diffraction frame
   Can be None, (dimx, dimy), or dim. In the latter case shape will be (dim, dim).

   *default* = ``None``

.. py:data:: .scan.data.save(str)

   *(35)* Saving mode

   Mode to use to save data to file.
   None: 
   'merge': attemts to merge data in single chunk [not implemented]
   'append': appends each chunk in master *.ptyd file
   'link': appends external links in master *.ptyd file and stores chunk separately in the path to the link. Links are relative file paths

   *default* = ``None``

.. py:data:: .scan.data.center(tuple)

   *(36)* Center (pixel) of the optical axes in data

   If None, this parameter will be set by 'auto_center' or elsewhere

   *default* = ``None``

.. py:data:: .scan.data.psize(float, tuple)

   *(37)* Detector pixel size

   Dimensions of the detector pixels (in meters)

   *default* = ``None (>0.0)``

.. py:data:: .scan.data.distance(float)

   *(38)* Sample-to-detector distance

   In meters.

   *default* = ``None (>0.0)``

.. py:data:: .scan.data.rebin(int)

   *(39)* Rebinning factor

   Rebinning factor for the raw data frames. 'None' or 1 both mean 'no binning'

   *default* = ``None (>1, <8)``

.. py:data:: .scan.data.orientation(int, tuple)

   *(40)* Data frame orientation

   None or 0: correct orientation
   1: invert columns (numpy.flip_lr)
   2: invert columns, invert rows
   3: invert rows  (numpy.flip_ud)
   4: transpose (numpy.transpose)
   4+i: tranpose + other operations from above
   
   Alternative a 3-tuple of booleans may be provided (do_transpose, do_flipud, do_fliplr)

   *default* = ``None``

.. py:data:: .scan.data.energy(float)

   *(41)* Photon energy of the incident radiation

   

   *default* = ``None (>0.0)``

.. py:data:: .scan.data.min_frames(int)

   *(42)* Minimum number of frames loaded by each node

   

   *default* = ``1``

.. py:data:: .scan.data.num_frames(int)

   *(43)* Maximum number of frames to be prepared

   If `positions_theory` are provided, num_frames will be ovverriden with the number of positions available

   *default* = ``None``

.. py:data:: .scan.data.chunk_format(str)

   *(44)* Appendix to saved files if save == 'link'

   

   *default* = ``.chunk%02d``

.. py:data:: .scan.data.auto_center(bool)

   *(45)* Determine if center in data is calculated automatically

   False: no automatic center 
   None: only if center is None 
   True: it will be enforced

   *default* = ``None``

.. py:data:: .scan.data.load_parallel(str)

   *(46)* Determines what will be loaded in parallel

   Choose among [None, 'data', 'common', 'all']

   *default* = ``data``

.. py:data:: .scan.data.positions_theory(ndarray)

   *(47)* Theoretical positions for this scan

   If provided, experimental positions from child class will be ignored. If data preparation is called from Ptycho instance, the calculated positions from the `pattern` dict will be inserted here

   *default* = ``None``

.. py:data:: .scan.data.experimentID(str)

   *(48)* Name of the experiment

   If None, a default value will be provided by the recipe.

   *default* = ``None``

.. py:data:: .scan.data.simulation(Param)

   *(49)* Simulated data as a preparation

   Similar to scan, simulation takes Parameters trees in the same form `illumination`, `sample` and `xy`. Any item in these trees will take precedence over scan specific parameters in the simulated scan.

   *default* = ``None``

.. py:data:: .scan.data.simulation.detector(Param, str, NoneType)

   *(50)* Detector parameters

   Can also be `None` if no detector specific filter is wanted or a string that matches one of the templates in the detector module

   *default* = ``None``

.. py:data:: .scan.data.simulation.psf(float)

   *(51)* Gaussian point spread with FWHM of `psf`

   `None` means no point spread

   *default* = ``None``

.. py:data:: .scan.sharing(Param)

   *(52)* Scan sharing options

   

   *default* = ``None``

.. py:data:: .scan.sharing.object_share_with(str)

   *(53)* Label or index of scan to share object with.

   Possible values:
   
   None: Do not share
   (string): label of the scan to share with
   (int):index of scan to share with

   *default* = ``None``

.. py:data:: .scan.sharing.object_share_power(float)

   *(54)* Relative power for object sharing

   

   *default* = ``1 (>0.0)``

.. py:data:: .scan.sharing.probe_share_with(str)

   *(55)* Label or index of scan to share probe with.

   Possible values:
   
   None: Do not share
   (string): label of the scan to share with
   (int):index of scan to share with

   *default* = ``None``

.. py:data:: .scan.sharing.probe_share_power(float)

   *(56)* Relative power for probe sharing

   

   *default* = ``1 (>0.0)``

.. py:data:: .scan.geometry(Param)

   *(57)* Physical parameters

   All distances are in meters. Other units are specified in the documentation strings.

   *default* = ``None``

.. py:data:: .scan.geometry.energy(float)

   *(58)* Energy (in keV)

   If None, use the wavelength instead.

   *default* = ``6.2 (>0.0)``

.. py:data:: .scan.geometry.lam(float)

   *(59)* Wavelength

   Used only if energy is None

   *default* = ``None (>0.0)``

.. py:data:: .scan.geometry.distance(float)

   *(60)* Distance from object to detector

   

   *default* = ``7.19 (>0.0)``

.. py:data:: .scan.geometry.psize(float)

   *(61)* Pixel size in Detector plane

   

   *default* = ``0.000172 (>0.0)``

.. py:data:: .scan.geometry.resolution(float)

   *(62)* Pixel size in Sample plane

   This parameter is used only for simulations

   *default* = ``None (>0.0)``

.. py:data:: .scan.geometry.prop_type(str)

   *(63)* Propagation type

   Either "farfield" or "nearfield"

   *default* = ``farfield``

.. py:data:: .scan.xy(Param)

   *(64)* Parameters for scan patterns

   These parameters are useful in two cases:
   
    - When the experimental positions are not known (no encoders)
    - When using the package to simulate data.

   *default* = ``None``

.. py:data:: .scan.xy.type(str)

   *(65)* Pattern type

   The type must be one of the following:
   
    - None (default): positions are read from data file.
    - 'custom': positions are read from parameter structure
    - 'raster': raster grid pattern
    - 'round': concentric circles pattern
    - 'round_roi': concentric circles cropped to a region-of-interest
    - 'spiral': spiral pattern
    - 'spiral_roi': spiral pattern cropped to a region-of-interest

   *default* = ``None (>0.0)``

.. py:data:: .scan.xy.positions(list)

   *(66)* List of positions

   This list is to be used if scan_type == 'custom' no proper positions are found in the prepared data file.

   *default* = ``None``

.. py:data:: .scan.xy.raster(Param)

   *(67)* Raster pattern parameters

   

   *default* = ``None``

.. py:data:: .scan.xy.raster.nx(int)

   *(68)* Number of steps in x

   

   *default* = ``10 (>0)``

.. py:data:: .scan.xy.raster.ny(int)

   *(69)* Number of steps in y

   

   *default* = ``10 (>0)``

.. py:data:: .scan.xy.raster.dx(float)

   *(70)* Step size (grid spacing)

   

   *default* = ``1.00E-06 (>0.0)``

.. py:data:: .scan.xy.raster.dy(float)

   *(71)* Step size (grid spacing)

   

   *default* = ``1.00E-06 (>0.0)``

.. py:data:: .scan.xy.round(Param)

   *(72)* Round pattern parameters

   

   *default* = ``None``

.. py:data:: .scan.xy.round.dr(float)

   *(73)* Spacing of concentric circles

   

   *default* = ``3.00E-07 (>0.0)``

.. py:data:: .scan.xy.round.nr(int)

   *(74)* Number of radial steps (number of circles - 1)

   

   *default* = ``10 (>0)``

.. py:data:: .scan.xy.round.nth(int)

   *(75)* Number of points in the inner circle

   

   *default* = ``5 (>0)``

.. py:data:: .scan.xy.round_roi(Param)

   *(76)* Round - ROI pattern parameters

   

   *default* = ``None``

.. py:data:: .scan.xy.round_roi.dr(float)

   *(77)* Spacing of concentric circles

   

   *default* = ``3.00E-07 (>0.0)``

.. py:data:: .scan.xy.round_roi.nth(int)

   *(78)* Number of points in the inner circle

   

   *default* = ``5 (>0)``

.. py:data:: .scan.xy.round_roi.lx(float)

   *(79)* ROI dimension in x

   

   *default* = ``3.00E-06 (>0.0)``

.. py:data:: .scan.xy.round_roi.ly(float)

   *(80)* ROI dimension in y

   

   *default* = ``3.00E-06 (>0.0)``

.. py:data:: .scan.xy.spiral(Param)

   *(81)* Spiral scan parameters

   

   *default* = ``None``

.. py:data:: .scan.xy.spiral.dr(float)

   *(82)* Spiral arm spacing

   

   *default* = ``2.00E-06 (>0.0)``

.. py:data:: .scan.xy.spiral.r_out(float)

   *(83)* Outer radius

   

   *default* = ``3.00E-05 (>0.0)``

.. py:data:: .scan.xy.spiral_roi(Param)

   *(84)* Spiral - ROI pattern parameters

   

   *default* = ``None``

.. py:data:: .scan.xy.spiral_roi.dr(float)

   *(85)* Spiral arm spacing

   

   *default* = ``2.00E-06 (>0.0)``

.. py:data:: .scan.xy.spiral_roi.lx(float)

   *(86)* ROI dimension in x

   

   *default* = ``2.00E-05 (>0.0)``

.. py:data:: .scan.xy.spiral_roi.ly(float)

   *(87)* ROI dimension in y

   

   *default* = ``2.00E-05 (>0.0)``

.. py:data:: .scan.illumination(Param)

   *(88)* Illumination model (probe)

   

   *default* = ``None (>0.0)``

.. py:data:: .scan.illumination.model(str)

   *(89)* Type of illumination model

   One of:
   `None` : model initialitziation defaults to flat array filled with the specified number of photons
   `recon` : load model from previous reconstruction, see `recon` Parameters
   `stxm` : Estimate model from autocorrelation of mean diffraction data
   `<resource>` : one of ptypys internal image resource strings
   `<template>` : one of the templates inillumination module
   In script, you may pass a numpy.array here directly as the model

   *default* = ``None``

.. py:data:: .scan.illumination.photons(int)

   *(90)* Number of photons in the incident illumination

   

   *default* = ``None (>0)``

.. py:data:: .scan.illumination.recon(Param)

   *(91)* Parameters to load from previous reconstruction

   

   *default* = ``None``

.. py:data:: .scan.illumination.recon.rfile(file)

   *(92)* Path to a .ptyr compatible file

   

   *default* = ``*.ptyr``

.. py:data:: .scan.illumination.recon.ID(NoneType)

   *(93)* ID (label) of storage data to load

   `None` is any ID

   *default* = ``None``

.. py:data:: .scan.illumination.recon.layer(float)

   *(94)* Layer (mode) of storage data to load

   `None` is all layers, choose `0` for main mode

   *default* = ``None``

.. py:data:: .scan.illumination.stxm(Param)

   *(95)* Parameters to initialize illumination from diffraction data

   

   *default* = ``None``

.. py:data:: .scan.illumination.stxm.label(str)

   *(96)* Scan label of diffraction that is to be used for probe estimate

   `None`, own scan label is used

   *default* = ``None``

.. py:data:: .scan.illumination.aperture(Param)

   *(97)* Beam aperture parameters

   

   *default* = ``None``

.. py:data:: .scan.illumination.aperture.form(str)

   *(98)* One of None, 'rect' or 'circ'

   One of:
   
    - None: no aperture, this may be useful for nearfield
    - 'rect': rectangular aperture
    - 'circ': circular aperture

   *default* = ``circ (>0.0)``

.. py:data:: .scan.illumination.aperture.diffuser(float)

   *(99)* Noise in the transparen part of the aperture

   Can be either:
   
   - None : no noise
   - 2-tuple: noise in phase (amplitude (rms), minimum feature size)
   - 4-tuple: noise in phase & modulus (rms, mfs, rms_mod, mfs_mod)

   *default* = ``None (>0.0)``

.. py:data:: .scan.illumination.aperture.size(float)

   *(100)* Aperture width or diameter

   May also be a tuple (vertical,horizontal) for size in case of an asymmetric aperture 

   *default* = ``None (>0.0)``

.. py:data:: .scan.illumination.aperture.edge(int)

   *(101)* Edge width of aperture (in pixels!)

   

   *default* = ``2 (>0)``

.. py:data:: .scan.illumination.aperture.central_stop(float)

   *(102)* size of central stop as a fraction of aperture.size

   If not None: places a central beam stop in aperture. The value given here is the fraction of the stop compared to size 

   *default* = ``None (>0.0, <1.0)``

.. py:data:: .scan.illumination.aperture.offset(float, tuple)

   *(103)* Offset between center of aperture and optical axes

   May also be a tuple (vertical,horizontal) for size in case of an asymmetric offset

   *default* = ``0``

.. py:data:: .scan.illumination.propagation(Param)

   *(104)* Parameters for propagation after aperture plane

   Propagation to focus takes precedence to parallel propagation if foccused is not None

   *default* = ``None``

.. py:data:: .scan.illumination.propagation.parallel(float)

   *(105)* Parallel propagation distance

   If None or 0 : No parallel propagation 

   *default* = ``None``

.. py:data:: .scan.illumination.propagation.focussed(float)

   *(106)* Propagation distance from aperture to focus

   If None or 0 : No focus propagation 

   *default* = ``None``

.. py:data:: .scan.illumination.propagation.antialiasing(float)

   *(107)* Antialiasing factor

   Antialiasing factor used when generating the probe. (numbers larger than 2 or 3 are memory hungry)

   *default* = ``1``

.. py:data:: .scan.illumination.propagation.spot_size(float)

   *(108)* Focal spot diameter

   If not None, this parameter is used to generate the appropriate aperture size instead of aperture.size

   *default* = ``None (>0.0)``

.. py:data:: .scan.illumination.diversity(Param)

   *(109)* Probe mode(s) diversity parameters

   Can be `None` i.e. no diversity

   *default* = ``None``

.. py:data:: .scan.illumination.diversity.noise(tuple)

   *(110)* Noise in the generated modes of the illumination 

   Can be either:
   
   - None : no noise
   - 2-tuple: noise in phase (amplitude (rms), minimum feature size)
   - 4-tuple: noise in phase & modulus (rms, mfs, rms_mod, mfs_mod)

   *default* = ``None``

.. py:data:: .scan.illumination.diversity.power(tuple, float)

   *(111)* Power of modes relative to main mode (zero-layer)

   

   *default* = ``0.1``

.. py:data:: .scan.illumination.diversity.shift(float)

   *(112)* Lateral shift of modes relative to main mode

   [not implemented]

   *default* = ``None``

.. py:data:: .scan.sample(Param)

   *(113)* Initial object modelization parameters

   

   *default* = ``None (>0.0)``

.. py:data:: .scan.sample.model(str)

   *(114)* Type of initial object model

   One of:
   `None` : model initialitziation defaults to flat array filled `fill`
   `recon` : load model from STXM analysis of diffraction data
   `stxm` : Estimate model from autocorrelation of mean diffraction data
   `<resource>` : one of ptypys internal model resource strings
   `<template>` : one of the templates in sample module
   In script, you may pass a numpy.array here directly as the model

   *default* = ``None``

.. py:data:: .scan.sample.fill(float)

   *(115)* Default fill value

   

   *default* = ``1``

.. py:data:: .scan.sample.recon(Param)

   *(116)* Parameters to load from previous reconstruction

   

   *default* = ``None``

.. py:data:: .scan.sample.recon.rfile(file)

   *(117)* Path to a .ptyr compatible file

   

   *default* = ``*.ptyr``

.. py:data:: .scan.sample.recon.ID(NoneType)

   *(118)* ID (label) of storage data to load

   `None` is any ID

   *default* = ``None``

.. py:data:: .scan.sample.recon.layer(float)

   *(119)* Layer (mode) of storage data to load

   `None` is all layers, choose `0` for main mode

   *default* = ``None``

.. py:data:: .scan.sample.stxm(Param)

   *(120)* STXM analysis parameters

   

   *default* = ``None``

.. py:data:: .scan.sample.stxm.label(str)

   *(121)* Scan label of diffraction that is to be used for probe estimate

   `None`, own scan label is used

   *default* = ``None``

.. py:data:: .scan.sample.process(Param)

   *(122)* Model processing parameters

   Can be `None`, i.e. no processing

   *default* = ``None``

.. py:data:: .scan.sample.process.offset(tuple)

   *(123)* Offset between center of object array and scan pattern

   

   *default* = ``(0,0) (>0.0)``

.. py:data:: .scan.sample.process.zoom(tuple)

   *(124)* Zoom value for object simulation.

   If None, leave the array untouched. Otherwise the modeled or loaded image will be resized using scipy.ndimage.zoom.

   *default* = ``None (>0.0)``

.. py:data:: .scan.sample.process.formula(str)

   *(125)* Chemical formula

   

   *default* = ``None``

.. py:data:: .scan.sample.process.density(float)

   *(126)* Density in [g/ccm]

   Only used if formula is not None

   *default* = ``1``

.. py:data:: .scan.sample.process.thickness(float)

   *(127)* Maximum thickness of sample

   if None, the absolute values of loaded source array will be used

   *default* = ``1.00E-06``

.. py:data:: .scan.sample.process.ref_index(complex)

   *(128)* Assigned refractive index

   If None, treat source array as projection of refractive index. If a refractive index is provided the array's absolute value will be used to scale the refractive index.

   *default* = ``0.5+0.j (>0.0)``

.. py:data:: .scan.sample.process.smoothing(int)

   *(129)* Smoothing scale

   Smooth the projection with gaussian kernel of width given by smoothing_mfs

   *default* = ``2 (>0)``

.. py:data:: .scan.sample.diversity(Param)

   *(130)* Probe mode(s) diversity parameters

   Can be `None` i.e. no diversity

   *default* = ``None``

.. py:data:: .scan.sample.diversity.noise(tuple)

   *(131)* Noise in the generated modes of the illumination 

   Can be either:
   
   - None : no noise
   - 2-tuple: noise in phase (amplitude (rms), minimum feature size)
   - 4-tuple: noise in phase & modulus (rms, mfs, rms_mod, mfs_mod)

   *default* = ``None``

.. py:data:: .scan.sample.diversity.power(tuple, float)

   *(132)* Power of modes relative to main mode (zero-layer)

   

   *default* = ``0.1``

.. py:data:: .scan.sample.diversity.shift(float)

   *(133)* Lateral shift of modes relative to main mode

   [not implemented]

   *default* = ``None``

.. py:data:: .scan.coherence(Param)

   *(134)* Coherence parameters

   

   *default* = ``None (>0.0)``

.. py:data:: .scan.coherence.num_probe_modes(int)

   *(135)* Number of probe modes

   

   *default* = ``1 (>0)``

.. py:data:: .scan.coherence.num_object_modes(int)

   *(136)* Number of object modes

   

   *default* = ``1 (>0)``

.. py:data:: .scan.coherence.spectrum(list)

   *(137)* Amplitude of relative energy bins if the probe modes have a different energy

   

   *default* = ``None (>0.0)``

.. py:data:: .scan.coherence.object_dispersion(str)

   *(138)* Energy dispersive response of the object

   One of:
   
    - None or 'achromatic': no dispersion
    - 'linear': linear response model
    - 'irregular': no assumption

   *default* = ``None``

.. py:data:: .scan.coherence.probe_dispersion(str)

   *(139)* Energy dispersive response of the probe

   One of:
   
    - None or 'achromatic': no dispersion
    - 'linear': linear response model
    - 'irregular': no assumption

   *default* = ``None``


.scans
------

.. py:data:: .scans(Param)

   *(140)* Container for instances of 'scan' parameters

   

   *default* = ``None``

.. py:data:: .scans.scan_%d(scan)

   *(141)* 

   

   *default* = ``None``


.engine
-------

.. py:data:: .engine(Param)

   *(142)* Reconstruction engine parameters

   

   *default* = ``None``

.. py:data:: .engine.common(Param)

   *(143)* Parameters common to all engines

   

   *default* = ``None``

.. py:data:: .engine.common.numiter(int)

   *(144)* Total number of iterations

   

   *default* = ``2000 (>0)``

.. py:data:: .engine.common.numiter_contiguous(int)

   *(145)* Number of iterations without interruption

   The engine will not return control to the caller until this number of iterations is completed (not processing server requests, I/O operations, ...)

   *default* = ``1 (>0)``

.. py:data:: .engine.common.probe_support(float)

   *(146)* Fraction of valid probe area (circular) in probe frame

   

   *default* = ``0.7 (>0.0)``

.. py:data:: .engine.common.clip_object(tuple)

   *(147)* Clip object amplitude into this intrervall

   

   *default* = ``None (>0.0)``

.. py:data:: .engine.DM(Param)

   *(148)* Parameters for Difference map engine

   

   *default* = ``None``

.. py:data:: .engine.DM.alpha(int)

   *(149)* Difference map parameter

   

   *default* = ``1 (>0)``

.. py:data:: .engine.DM.probe_update_start(int)

   *(150)* Number of iterations before probe update starts

   

   *default* = ``2 (>0)``

.. py:data:: .engine.DM.update_object_first(bool)

   *(151)* If False update object before probe

   

   *default* = ``TRUE (>0.0)``

.. py:data:: .engine.DM.overlap_converge_factor(float)

   *(152)* Threshold for interruption of the inner overlap loop

   The inner overlap loop refines the probe and the object simultaneously. This loop is escaped as soon as the overall change in probe, relative to the first iteration, is less than this value.

   *default* = ``0.05 (>0.0)``

.. py:data:: .engine.DM.overlap_max_iterations(int)

   *(153)* Maximum of iterations for the overlap constraint inner loop

   

   *default* = ``10 (>0)``

.. py:data:: .engine.DM.probe_inertia(float)

   *(154)* Weight of the current probe estimate in the update

   

   *default* = ``0.001 (>0.0)``

.. py:data:: .engine.DM.object_inertia(float)

   *(155)* Weight of the current object in the update

   

   *default* = ``0.1 (>0.0)``

.. py:data:: .engine.DM.fourier_relax_factor(float)

   *(156)* If rms error of model vs diffraction data is smaller than this fraction, Fourier constraint is met

   Set this value higher for noisy data

   *default* = ``0.01 (>0.0)``

.. py:data:: .engine.DM.obj_smooth_std(int)

   *(157)* Gaussian smoothing (pixel) of the current object prior to update

   If None, smoothing is deactivated. This smoothing can be used to reduce the amplitude of spurious pixels in the outer, least constrained areas of the object.

   *default* = ``20 (>0)``

.. py:data:: .engine.ML(Param)

   *(158)* Maximum Likelihood parameters

   

   *default* = ``None``

.. py:data:: .engine.ML.type(str)

   *(159)* Likelihood model. One of 'gaussian', 'poisson' or 'euclid'

   [only 'gaussian' is implemented for now]

   *default* = ``gaussian (>0.0)``

.. py:data:: .engine.ML.floating_intensities(bool)

   *(160)* If True, allow for adaptative rescaling of the diffraction pattern intensities (to correct for incident beam intensity fluctuations).

   

   *default* = ``FALSE``

.. py:data:: .engine.ML.intensity_renormalization(float)

   *(161)* A rescaling of the intensity so they can be interpreted as Poisson counts.

   

   *default* = ``1 (>0.0)``

.. py:data:: .engine.ML.reg_del2(bool)

   *(162)* Whether to use a Gaussian prior (smoothing) regularizer.

   

   *default* = ``TRUE (>0.0)``

.. py:data:: .engine.ML.reg_del2_amplitude(float)

   *(163)* Amplitude of the Gaussian prior if used.

   

   *default* = ``0.01 (>0.0)``

.. py:data:: .engine.ML.smooth_gradient(float)

   *(164)* Smoothing preconditioner. If 0, not used, if > 0 gaussian filter if < 0 Hann window.

   

   *default* = ``0 (>0.0)``

.. py:data:: .engine.ML.scale_precond(bool)

   *(165)* Whether to use the object/probe scaling preconditioner.

   This parameter can give faster convergence for weakly scattering samples.

   *default* = ``FALSE (>0.0)``

.. py:data:: .engine.ML.scale_probe_object(float)

   *(166)* Relative scale of probe to object.

   

   *default* = ``1 (>0.0)``

.. py:data:: .engine.ML.probe_update_start(int)

   *(167)* Number of iterations before probe update starts

   

   *default* = ``0``


.engines
--------

.. py:data:: .engines(Param)

   *(168)* Container for instances of "engine" parameters

   All engines registered in this structure will be executed sequentially.

   *default* = ``None``

.. py:data:: .engines.engine_%d(engine)

   *(169)* 

   

   *default* = ``None``

