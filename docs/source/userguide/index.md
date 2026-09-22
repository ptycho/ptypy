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
generated/notebooks/basic_examples/00_the_parameter_tree.ipynb
generated/notebooks/basic_examples/01_using_yaml_or_json_config.ipynb
generated/notebooks/basic_examples/02_input_output_parameters.ipynb
generated/notebooks/basic_examples/03_scan_models.ipynb
setting_probe_init.md
generated/legacy/rst/concept.rst
generated/legacy/rst/data_management.rst
```

## Core Reconstruction Engines
PtyPy offers a range of different core reconstruction engines that can be grouped into the 3 main categories of
projectional (DM, RAAR), stochastic (ePIE, SDR) and gradient-based (ML) engines. All of these engines are
available as parallel CPU engines (MPI) and accelerated GPU engines (Cupy/PyCuda + MPI).
The following articles describe the core engines and their features in more detail.

```{toctree}
:maxdepth: 1
:caption: Core Engine
generated/notebooks/basic_examples/04_choosing_engines.ipynb
generated/notebooks/basic_examples/05_projectional_engines.ipynb
generated/notebooks/basic_examples/06_stochastic_engines.ipynb
generated/notebooks/basic_examples/07_gradient_based_engines.ipynb
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

## Quality and Metrics

```{toctree}
:maxdepth: 1
:caption: Metrics
reporting.md
```

