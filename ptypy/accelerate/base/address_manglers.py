'''
utils to help with position refinement
'''

import numpy as np
np.random.seed(0)
class RandomIntMangle(object):
    '''
    assumes integer pixel shift.
    '''
    def __init__(self, max_step_per_shift,  start, stop, max_bound=None,  randomseed=None):
        # can be initialised in the engine.init

        self.max_bound = max_bound  # maximum distance from the starting positions
        self.max_step = lambda it: (max_step_per_shift * (stop - it) / (stop - start)) # maximum step per iteration, decreases with progression
        self.call_no = 0

    def mangle_address(self, addr_current, addr_original, iteration):
        '''
        Takes the current address book and adds an offset to it according to the parameters
        '''
        mangled_addr = np.zeros_like(addr_current)
        mangled_addr[:] = addr_current  # make a copy
        max_step = self.max_step(iteration)
        deltas = np.random.randint(0, max_step + 1, (addr_current.shape[0], 2))
        # the following improves things a lot!
        deltas[:, 0] *= (-1)**self.call_no 
        deltas[:, 1] *= (-1)**(self.call_no//2)
        self.call_no += 1

        # deltas = np.zeros((addr_current.shape[0], 2)) # for testing
        old_positions = np.zeros((addr_current.shape[0], 2))
        old_positions[:] = addr_current[:, 0, 1, 1:]
        new_positions = np.zeros((addr_current.shape[0],2))
        # new_positions[1:] = old_positions[1:] + deltas[1:]  # first mode is same as all of them.
        new_positions[:] = old_positions + deltas  # first mode is same as all of them.
        self.apply_bounding_box(new_positions, old_positions, addr_original)
        #  now update the main matrix (Same for all modes)
        for idx in range(addr_original.shape[1]):
            mangled_addr[:, idx, 1, 1:] = new_positions
        return mangled_addr

    def apply_bounding_box(self, new_positions, old_positions, addr_original):
        '''
        Checks if the new co-ordinates lie within the bounding box. If not, we undo this move.
        '''

        distances_from_original = new_positions - addr_original[:, 0, 1, 1:]
        # logger.warning("distance from original is %s" % repr(distances_from_original))
        norms = np.linalg.norm(distances_from_original, axis=-1)
        for i in range(len(new_positions)):
            if norms[i]> self.max_bound:
                new_positions[i] = old_positions[i]
        # new_positions[norms>self.max_bound] = old_positions[norms>self.max_bound] # make sure we aren't outside the bounding box
#