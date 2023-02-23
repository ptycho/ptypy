# PtyPy 0.7.1 release notes

Patch release.

*  Bug fix in Numpy FFT propagator - enforcing C-contiguous arrays
*  You can now choose the CPU FFT type with parameter `p.scans.<scan_00>.ffttype={'scipy','numpy','fftw'}`


# PtyPy 0.7 release notes

This release is focused on improving the usability of PtyPy in Jupyter notebooks in preparation for the 
[PtyPy workshop](https://www.diamond.ac.uk/Home/Events/2023/Ptychography--PtyPy--Software-Workshop-2023.html) 
held at the Diamond Light Source in January 2023. The workshop features extensive interactive 
[tutorials](https://ptycho.github.io/tutorials) delivered using Jupyter notebooks. 

## Build changes

We added the following features

* convenience functions to read parameters from JSON/YAML files 
  (`ptypy.utils.param_from_json` and `ptypy.utils.param_from_yaml`)
* plotting utilites `ptypy.utils.plot_client.figure_from_ptycho` and 
  `ptypy.utils.plot_client.figure_from_ptyr` that can be useful in Jupyter notebooks
* non-threaded interactive plotting for Jupyter notebooks using `p.io.autoplot.threaded=False`


# PtyPy 0.6 release notes

Public release of ptypy! After having to use a private repository
for many years, we finally have a license agreement that allows
PtyPy to go public. 

## Build changes

In accordance with PEP 621, we are moving away 
from `setup.py` for the main build of PtyPy and
adopt the new community standard for building
packages with `pyproject.toml`.


# PtyPy 0.5 release notes

We're excited to bring you a new release, with new engines, GPU accelerations and
many smaller improvements.

## Engine Updates

### New abstraction layer for most engines, new engines.

 * generalised projectional engine with derived engines DM, RAAR
 * generalised stochastic engine with derived engines EPIE, SDR
 
Engines that are based on global projections now all derive from a generalized
base engine that is able to express most common projection algorithms with 4 scalar parameters.
DM and RAAR are two such derived classes. Similarly, algorithms based on a stochastic
sequence of local projections (SDR, EPIE) now inherit from a common base engine. 

### GPU acceleration

 * GPU-acceleration for all major engines DM, ML, EPIE, SDR, RAAR
 * accelerated engines needs to be imported **explicitly** with 
   ```python
   import ptypy
   ptypy.load_gpu_engines('cuda')
   ```
 
We accelerated three engines (projectional, stochastic and ML) using
the [`PyCUDA`](https://documen.tician.de/pycuda/) and 
[`Reikna`](http://reikna.publicfields.net/en/latest/) library and a whole
collection of custom kernels. 

All GPU engines leverage a "streaming" model which means that the 
primary locations of all objects are on the host (CPU) memory.
Diffraction data arrays and all other arrys that scale linearly with 
the number of shifts/positons are segmented into blocks (of frames).
The idea is that these blocks are moved on and off the device (GPU) during
engine iteration if the GPU does not have enough memory to store all
blocks. The number of frames per block can
be adjusted with the new top-level
[`frames_per_block`](https://ptycho.github.io/ptypy/rst/parameters.html#ptycho.frames_per_block)
parameter. This parameter has little influence for smaller problem size,
but needs to be adjusted if your GPU has too little memory to fit even
a single block. 

Each engine iteration will cycle through all blocks, DM needs to even cycle 
once for each projection. We therefore recommend to make the block size small 
enough such that at least a couple of blocks fit on the GPU to hide the latency of 
data transfers. For best 
performance, we employ a mirror scheme such that each cycle reverses the 
block order and reduces the host to device copies (and vice versa) to the
absolute minimum.

GPU engines work in parallel when each MPI rank takes one GPU. For sending
data between ranks, PtyPy will perform a host copy first in most cases or
use whatever the underlying MPI implementations does for CUDA-aware MPI
(only tested for OpenMPI). Unfortunately, this mapping of one rank per 
GPU will leave CPU cores idle if there are more cores on the system than GPUs. 

Within a node, PtyPy can use nccl (requires a CuPy install 
and setting `PTYPY_USE_NCCL=1`) for passing data between ranks/GPUs.


## Breaking changes

### Ptyscan classes (experiment) need to be imported explicitly

Most derived PtyScan classes (all those in the `/experiment` folder) now need
to be imported explicitly. We took this step to separate the user space
more clearly from the base package and to avoid dependency creep from
user-introduced code. At the beginning of your script, you now 
need to import your module explicitly or use one of the helper 
functions.

```python
import ptypy
ptypy.load_ptyscan_module(module='')
ptypy.load_all_ptyscan_modules()
```

Any PtyScan derived class in these modules that is decorated 
with the `ptypy.experiment.register()` function will now be included 
in the parameter tree and selectable by name.

If you prefer the old way of importing ptypy "fully loaded", just use
```python
import ptypy
ptypy.load_all()
```
which attempts to load all optional PtyScan classes and all engines.


## Other updates

 1. Code for `utils.parallel.bcast_dict` and `gather_dict` has been simplified and
    should be backwards compatible.
 2. The `fourier_power_bound` that was previously calculated internally from
    the `fourier_relax_factor` can now be set explicitly and we recommend that from
    now on. The recommended value for the`fourier_power_bound` is 0.25 for Poisson statistics
    (see supplementary of [`this paper`](https://www.pnas.org/doi/10.1073/pnas.0905846107#supplementary-materials))
 3. Position correction now supports an alternate search scheme, i.e. along a fixed grid.
    This scheme is more accurate than a stochastic search and the overhead incurred
    for this brute force search is acceptable for GPU engines.
 4. We switched to a pip install within a conda environment as the main supported way of installation
 
## Roadmap

 * Automatic adjustment of the block sizes.
 * Improve scaling behavior across multiple nodes and high frame counts.
 * Better support for live processing (on a continuous detector data stream).
 * More tests.
 * Branch cleaning.
 
## Contributors

Thanks to the efforts at the Diamond Light Source that made this
update possible.

 * Aaron Parsons
 * Bjoern Enders
 * Benedikt Daurer
 * Joerg Lotze
 

# PtyPy 0.4 release notes

After quite some work we announce ptypy 0.4. Apart from including all the fixes and improvements from 0.3.0 to 0.3.1, it includes two bigger changes
 1. Ptypy has now been converted to python 3 and will be **python 3 only** in future. The python 2 version will not be actively maintained anymore, we keep a branch for it for a while but we don't expect to put in many fixes and certainly not any new features. Team work by Julio, Alex, Bjoern and Aaron.
 *Please note: all branches that haven’t been converted to python 3 by the end of 2019 will most likely be removed during 2020.* Please rebase your effort on version 0.4. If you need help rebasing your efforts, please let us know soon.
 2. Position correction is now supported in most engines. It has been implemented by Wilhelm Eschen following the annealing approach introduced by A.M. Maiden et al. (Ultramicroscopy, Volume 120, 2012, Pages 64-72). Bjoern, Benedikt and Aaron helped refine and test it.

## Roadmap

The next release will focus on scalability for HPC applications and GPU acceleration. 


# PtyPy 0.3 release notes

We are happy to announce that ptypy 0.3 is now out. If you have been using the ptypy 0.2 (from the master branch), the transition should be smooth but far from automatic - see below. The essence of this new release is
  1. a redesign of ptypy's internal structure, especially the introduction of an extendable [`ScanModel`](https://github.com/ptycho/ptypy/blob/master/ptypy/core/manager.py), which should make new ideas and new algorithms easier to implement (a big collective effort involving A. Björling, A. Parsons, B. Enders and P. Thibault),
  2. support for 3D Bragg ptychography, which uses the new `ScanModel` structure (all thanks to A. Björling),

  3. extensive testing of most components of the code, and Travis CI integration (huge work by A. Parsons and important contributions by S. Chalkidis), 

  4. more dimensions for `Storage` classes, reduced memory footprint and reduced object count, as `Views` are now slotted and don't hold other objects (B. Enders and A. Björling), and
  
  5. the introduction of the [`descriptor`](https://github.com/ptycho/ptypy/blob/master/ptypy/utils/descriptor.py) submodule, which manages the whole parameter tree, including validation, defaults, and documentation (collective effort led by B. Enders and P. Thibault)
   

## Breaking changes

The streamlining of the input parameters means that *all reconstruction scripts for version 0.2 will now fail*. We had no choice.

The changes were required in order to solve the following problems:
  1. Parameter definitions, documentations and defaults were in different locations, so hard to track and maintain
  2. The meaning of a branch set to `None` was ambiguous.
  3. Basic experiment geometry (some distances, radiation energy, etc) could be specified at two different locations.
  4. In general, the standards were not clear.

The solution to all these problems came with the `descriptor` submodule. For a user, what matters most is that `ptypy.defaults_tree` now contains the description of the full set of parameters known to ptypy.

### `defaults_tree`

Here's a short example of how `defaults_tree` is used internally, and how you can used it in your scripts or on the command line to inspect `ptypy`'s parameter structure.

```python
import ptypy

# Extract one branch
desc_DM_simple = ptypy.defaults_tree['engine.DM_simple']

# Print out the description of all sub-parameters
print desc_DM_simple.to_string()
```
```
[numiter]
lowlim = 1
help = Total number of iterations
default = 123
type = int

[numiter_contiguous]
lowlim = 1
help = Number of iterations without interruption
default = 1
doc = The engine will not return control to the caller until this number of iterations is completed (not processing server requests, I/O operations, ...).
type = int

[probe_support]
lowlim = 0.0
help = Valid probe area as fraction of the probe frame
default = 0.7
doc = Defines a circular area centered on the probe frame, in which the probe is allowed to be nonzero.
type = float

[name]
help = 
default = DM_simple
doc = 
type = str

[alpha]
lowlim = 0.0
help = Difference map parameter
default = 1
type = float

[overlap_converge_factor]
lowlim = 0.0
help = Threshold for interruption of the inner overlap loop
default = 0.05
doc = The inner overlap loop refines the probe and the object simultaneously. This loop is escaped as soon as the overall change in probe, relative to the first iteration, is less than this value.
type = float

[overlap_max_iterations]
lowlim = 1
help = Maximum of iterations for the overlap constraint inner loop
default = 10
type = int
```
```python
# Generate defaults
p = desc_DM_simple.make_default(depth=1)

# Validate
# (try with ptypy.utils.verbose.set_level(5) to get detailed DEBUG output)
desc_DM_simple.validate(p)

# Here's what happens if a parameter is wrong:
p.numiter = 'a'
desc_DM_simple.validate(p)
```
```
ERROR root - numiter                                            type                 INVALID
(...)
```

### How reconstruction scripts have to be changed

1. All `scans` sub-entry have a `name`. This name is one of the `ScanModel` classes, for now only `Vanilla`, `Full`, and `Bragg3dModel`. Most users will want to use `Full`. Others will come as we implement engines that require fundamental changes in the `pod` creation.

2. Data preparation: the sub-tree `recipe` does not exist anymore, and all parameters associated to a `PtyScan` subclass are specified directly in the `scan.???.data` sub-tree. The `geometry` sub-tree is also gone, with all relevant parameters also in the `scan.???.data` sub-tree.

3. There is no more an `.engine` sub-tree. This used to be present to change the default parameters of specific engines (or all of them using `engine.common`) before engines are declared in the `engines.*` sub-tree. We have found that this duplication is a source of confusion. Now the single place where engine parameters are set are within the `engines.*` sub-tree.

4. A sub-tree cannot be set to `None`. To deactivate a feature associated to a sub-tree, one has to set `active=False`. For instance `io.autoplot = None` is not valid anymore, and `io.autoplot.active = False` has to be used instead.

## Other contributions

 * Option to use pyfftw (thanks to L. Bloch, ESRF)
 * Scalability tests (thanks to C. Kewish, Australian Synchrotron)
 * A first draft jupyter-based plot client (B. Daurer, now National University of Singapore)
 * Bug fixes and tests (many people)

## Roadmap

The next release will focus on optimisation and speed. We will also soon switch to python 3.
