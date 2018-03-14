'''
object_probe_interaction

Contains things pertinent to the probe and object interaction.
Should have all the engine updates
'''

import numpy as np
from . import COMPLEX_TYPE

def difference_map_realspace_constraint(probe_and_object, exit_wave, alpha):
    raise NotImplementedError("This method is not implemented yet.")

def scan_and_multiply(probe, obj, exit_shape, addresses):
    raise NotImplementedError("This method is not implemented yet.")

def difference_map_update_object(ob, object_weights, probe, exit_wave, addr_info, cfact_object, ob_smooth_std=None, clip_object=None):
    raise NotImplementedError("This method is not implemented yet.")

def difference_map_update_probe(ob, probe_weights, probe, exit_wave, addr_info, cfact_probe, probe_support=None):
    raise NotImplementedError("This method is not implemented yet.")

def extract_array_from_exit_wave(exit_wave, exit_addr, array_to_be_extracted, extract_addr, array_to_be_updated, update_addr, cfact, weights):
    raise NotImplementedError("This method is not implemented yet.")

def center_probe(probe, center_tolerance):
    raise NotImplementedError("This method is not implemented yet.")

