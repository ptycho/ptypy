'''
useful utilities from ptypy that should be ported to gpu. These don't ahve external dependencies
'''
import numpy as np
from . import COMPLEX_TYPE
from .gpu_extension import abs2, sum_to_buffer, sum_to_buffer_stride, \
  norm2, mass_center, clip_complex_magnitudes_to_range, interpolated_shift, \
  complex_gaussian_filter
