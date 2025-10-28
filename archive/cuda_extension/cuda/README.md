# CUDA Module for Ptypy

Builds the functions for GPU extension module, using CMake 3.8+.

It gets built automatically together with setup.py.

## Tests

Some unit tests are implemented on the C++ level, using google
test. The test framework is automatically downloaded and built
during the regular cmake build. Run "make test" in the build 
directory (normally build/cuda) to run the tests.

## Naming Conventions

- parameter_name_ ==  this->parameter_name
- d_parameter_name == device pointer to parameter, everything else is host

## setup.py Build
