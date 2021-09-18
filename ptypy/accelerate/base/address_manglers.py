'''
utils to help with position refinement
'''

import numpy as np
np.random.seed(0)

class BaseMangler(object):
    '''
    Assumes integer pixel shift.
    '''
    def __init__(self, max_step_per_shift,  start, stop, nshifts, decay=True, max_bound=None,  randomseed=None):
        # can be initialised in the engine.init

        # maximum distance from the starting positions
        self.max_bound = max_bound  
        # maximum step per iteration, decreases with progression
        self.max_step = lambda it: np.ceil(max_step_per_shift * (stop - it) / (stop - start)) if decay else max_step_per_shift 
        self.nshifts = nshifts
        self.delta = 0

    def get_address(self, index, addr_current, mangled_addr, max_oby, max_obx):
        '''
        Mangles with the address given a delta shift
        '''
        old_positions = np.zeros((addr_current.shape[0], 2))
        old_positions[:] = addr_current[:, 0, 1, 1:]
        new_positions = np.zeros((addr_current.shape[0],2))
        new_positions[:] = old_positions + self.delta[index]  # first mode is same as all of them.
        #  now update the main matrix (Same for all modes)
        for idx in range(addr_current.shape[1]):
            mangled_addr[:, idx, 1, 1:] = new_positions
        self.apply_bounding_box(mangled_addr[:,:,1,1], 0, max_oby)
        self.apply_bounding_box(mangled_addr[:,:,1,2], 0, max_obx)

    def apply_bounding_box(self, addr, min, max):
        '''
        Check if the mangled addresses are within valid bounds
        '''
        addr[addr<min] = min
        addr[addr>max] = max

    def setup_shifts(self, current_iteration, nframes=1):
        ''' 
        Arrange an array of shifts
        '''
        raise NotImplementedError("This method needs to be overwritten in order to position correct")


class RandomIntMangler(BaseMangler):

    def __init__(self, *args, **kwargs):
        super(RandomIntMangler, self).__init__(*args, **kwargs)

    def setup_shifts(self, current_iteration, nframes=1):
        '''
        Calculates random integer shifts
        '''
        max_step = self.max_step(current_iteration)
        self.delta = np.random.randint(0, max_step + 1, (self.nshifts, nframes, 2))
        for index in range(self.nshifts):
            self.delta[index, :, 0] *= (-1)**index 
            self.delta[index, :, 1] *= (-1)**(index//2)
        # check if the shifts are within the maximum bound
        norms = np.linalg.norm(self.delta, axis=-1)
        self.delta[norms > self.max_bound] = 0

class GridSearchMangler(BaseMangler):
    def __init__(self, *args, **kwargs):
        super(GridSearchMangler, self).__init__(*args, **kwargs)

    def setup_shifts(self, current_iteration, nframes=1):
        '''
        Calculates integer shifts on a grid
        '''
        max_step = self.max_step(current_iteration)
        delta = np.mgrid[-max_step:max_step+1:1,
                         -max_step:max_step+1:1]
        within_bound = (delta[0]**2 + delta[1]**2) < (self.max_bound**2)
        self.delta = np.tile(delta[:,within_bound].T.reshape(within_bound.sum(),1,2), (1,nframes,1))
        self.nshifts = self.delta.shape[0]


        



