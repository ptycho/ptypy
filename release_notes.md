# PtyPy 0.4 release notes

After quite some work we announce ptypy 0.4. Apart from including all the fixes and improvements from 0.3.0 to 0.3.1, it includes two bigger changes
 1. Ptypy has now been converted to python 3 and will be **python 3 only** in future. The python 2 version will not be actively maintained anymore, we keep a branch for it for a while but we don't expect to put in many fixes and certainly not anny new features. Team work by Julio, Alex, Bjoern and Aaron.
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
