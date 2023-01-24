#pragma once

#ifndef PTYPY_CUPY_NVTRC
// pycuda code
#  include <thrust/complex.h>
using thrust::complex;

#else
// cupy code
#  include <cupy/complex.cuh>

#endif