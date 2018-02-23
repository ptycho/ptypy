'''
a module to holds the constraints
'''

import numpy as np

from error_metrics import log_likelihood, far_field_error, realspace_error
from object_probe_interaction import difference_map_realspace_constraint, scan_and_multiply
from propagation import farfield_propagator
import array_utils as au
from . import COMPLEX_TYPE, FLOAT_TYPE

def renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound):
    raise NotImplementedError("This method is not implemented yet.")

def get_difference(addr_info, alpha, backpropagated_solution, err_fmag, exit_wave, pbound, probe_object):
    raise NotImplementedError("This method is not implemented yet.")

def difference_map_fourier_constraint(mask, Idata, obj, probe, exit_wave, addr, prefilter, postfilter, pbound=None, alpha=1.0, LL_error=True):
    raise NotImplementedError("This method is not implemented yet.")


