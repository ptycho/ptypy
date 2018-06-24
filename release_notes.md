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

The solution to all these problems came with the `descriptor` submodule. For a user, what matter most is that `ptypy.defaults_tree` now contains the description of the full set parameters known to ptypy. 

## Bug fixes

## Roadmap
