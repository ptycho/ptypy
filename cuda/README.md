CUDA Module for Ptypy
=====================

Builds the functions for GPU extension module, using CMake 3.8+.

It gets built automatically together with setup.py.

naming conventions:
- parameter_name_ ==  this->parameter_name
- d_parameter_name == device pointer to parameter, everything else is host

setup.py build:
