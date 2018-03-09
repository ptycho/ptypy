'''
A module of the relevant error metrics
'''

from propagation import farfield_propagator
from object_probe_interaction import scan_and_multiply
from . import FLOAT_TYPE
import numpy as np


def log_likelihood(probe_obj, mask, exit_wave, Idata, prefilter, postfilter, addr):
    raise NotImplementedError("This method is not implemented yet.")

def far_field_error(current_solution, measured_solution, mask):
    raise NotImplementedError("This method is not implemented yet.")

def realspace_error(difference_in_exitwave):
    raise NotImplementedError("This method is not implemented yet.")
