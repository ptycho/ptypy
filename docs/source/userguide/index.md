# User Guide

## Getting Started
```{toctree}
:maxdepth: 1
generated/legacy/rst/getting_started.rst
```

## The Basic Concepts
PtyPy is a Python-based reconstruction framework that offers a large variety of features which can be
configured using a "parameter tree". The following articles describe the basic concepts of PtyPy and
how to work with this parameter tree.

```{toctree}
:maxdepth: 1
:caption: Basics
parameters.md
config_files.md
input_output.md
scan_models.md
setting_probe_init.md
reporting.md
```

## Core Reconstruction Engines
PtyPy offers a range of different core reconstruction engines that can be grouped into the 3 main categories of
projectional (DM, RAAR), stochastic (ePIE, SDR) and gradient-based (ML) engines. All of these engines are
available as parallel CPU engines (MPI) and accelerated GPU engines (Cupy/PyCuda + MPI).
The following articles describe the core engines and their features in more detail.

```{toctree}
:maxdepth: 1
:caption: Core Engines
engine_overview.md
projectional.md
stochastic.md
gradient_based.md
```

## Custom Reconstruction Engines
Over the years, the PtyPy user community has added a range of new or modified reconstruction engines
that provide extra functionality but are not supported and maintained by the PtyPy development team.
These engines need to specifically be imported from ```ptypy.custom``` when used in a PtyPy run script.
The following articles describe how these custom engines can be used and what features they provide.

```{toctree}
:maxdepth: 1
:caption: Custom Engines
multislice.md
indep_probes.md
lbfgs.md
wasp.md
object_regul.md
```
