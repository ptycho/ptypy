# Ptypy 0.3 release notes

We are happy to announce that ptypy 0.3 is now out. If you have been using the ptypy 0.2 (from the master branch), the transition should be smooth but far from automatic - see below "Breaking changes". The essence of this new release is a redesign of ptypy's internal structure, especially the introduction of an extendable [`ScanModel`](https://github.com/ptycho/ptypy/blob/master/ptypy/core/manager.py), which should make new ideas and new algorithms easier to implement, and the introduction of the [`descriptor`](https://github.com/ptycho/ptypy/blob/master/ptypy/utils/descriptor.py) submodule, which manages the whole parameter tree (including validation, defaults, and documentation).

## New features



## Improvements



## Breaking changes

The streamlining of the input parameters means that *all reconstruction scripts for version 0.2 will now fail*. We had no choice.

The changes were needed to solve the following problems:
  1. Parameter definitions, documentations and defaults were in different locations, so hard to track and maintain
  2. The meaning of a branch set to `None` was ambiguous.
  3. In general, the standards were not clear.

The solution to all these problems came with the `descriptor` submodule. For a user, what matter most is that `ptypy.defaults_tree` now contains the description of the full set of parameters known to ptypy. 

### `defaults_tree`

Here's a short example of how `defaults_tree` is used internally, and how you can used it in your script or on the command line to inspect `ptypy`'s parameter structure.

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

# Here's what happen if a parameter is wrong:
p.numiter = 'a'
desc_DM_simple.validate(p)
```
```
ERROR root - numiter                                            type                 INVALID
(...)
```

### How reconstructions script have to be changed

TODO: show an example of the most important changes in the parameter structure.


## Bug fixes

TODO: summarize the most important bugs that have been fixed.

## Roadmap

1. GPU acceleration
2. New features (OPR, position refinement, ...)
3. ...
