# -*- coding: utf-8 -*-
"""\
Data preparation for the I08 beamline, Diamond Light Source.

Written by A. Parsons, V.C.S. Kuppili and P. Thibault, July 2015.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.

"""
import numpy as np
import time
import os

if True: #__name__ == "__main__":
    import ptypy
    from ptypy import utils as u
    from ptypy import io
    from ptypy.utils.verbose import log
    from ptypy.core.paths import Paths
    # FIXME: Accessing the real "io" from the parent Ptycho class would be much better
    from ptypy.core import DEFAULT_io as IO_par
else:
    from .. import utils as u
    from .. import io
    from ..core.data import PtyScan
    from ..utils.verbose import log
    from ..core.paths import Paths
    from ..core import DEFAULT_io as IO_par


# Parameters for the nexus file saved by GDA
NXS_PATHS = u.Param()
NXS_PATHS.frame_pattern='entry1/instrument/_andorrastor/data'
FLAT_PATHS = u.Param()
FLAT_PATHS.key = "flat"
# Parameters for the hdf5 file saved by the STXM software
STXM_PATHS = u.Param()
STXM_PATHS.motors = 'entry1/Counter1/'
STXM_PATHS.energy = 'entry1/Counter1/'

# I08 recipe default parameters
RECIPE = u.Param()
RECIPE.base_path = None
RECIPE.scan_number = None
RECIPE.scan_number_stxm = None
RECIPE.dark_number = None
RECIPE.dark_number_stxm = None
RECIPE.dark_value = 200.   # Used if dark_number is None
RECIPE.detector_flat_file = None
RECIPE.nxs_file_pattern = '%(base_path)s/nexus/i08-%(scan_number)s.nxs'
RECIPE.dark_nxs_file_pattern = '%(base_path)s/nexus/i08-%(dark_number)s.nxs'
RECIPE.date = None 
RECIPE.stxm_file_pattern = '%(base_path)s/%(date)s/discard/Sample_Image_%(date)s_%(scan_number_stxm)s.hdf5'


RECIPE.motors = ['sample_y','sample_x'] # same orientation as I13 for now
RECIPE.energy = None
RECIPE.lam=None
RECIPE.z = None
RECIPE.motors_multiplier = 1e-6

# Default generic parameter set from
I08DEFAULT = ptypy.core.data.PtyScan.DEFAULT.copy()

I08DEFAULT.recipe = RECIPE

I08DEFAULT.auto_center = False


class I08Scan(ptypy.core.data.PtyScan):
    DEFAULT = I08DEFAULT
    
    def __init__(self, pars=None, **kwargs):
        """
        I08 (Diamond Light Source) data preparation class.
        """
        # Initialize parent class. All updated parameters are now in
        # self.info
        RDEFAULT = RECIPE.copy()
        RDEFAULT.update(pars.recipe)
        pars.recipe.update(RDEFAULT)
        
        super(I08Scan, self).__init__(pars, **kwargs)

        # Try to extract base_path to access data files
        if self.info.recipe.base_path is None:
            d = os.getcwd()
            base_path = None
            while True:
                if 'raw' in os.listdir(d):
                    base_path = d
                    break
                d, rest = os.path.split(d)
                if not rest:
                    break
            if base_path is None:
                raise RuntimeError('Could not guess base_path.')
            else:
                self.info.recipe.base_path = base_path

        # Sanity check: for now we need a date to identify the SXTM file
        if self.info.recipe.date is None:
            raise RuntimeError('recipe.date has to be specified to find the STXM file name.')
        else:
            try: 
                time.strptime(self.info.recipe.date, '%Y-%m-%d') 
            except ValueError:
                print('The date should be in format "YYYY-MM-DD"')
                raise
        
        # Construct the file names
        self.nxs_filename = self.info.recipe.nxs_file_pattern % self.info.recipe
        self.stxm_filename = self.info.recipe.stxm_file_pattern % self.info.recipe
        log(3, 'Will read from nxs file %s' % self.nxs_filename)
        log(3, 'Will read from STXM file %s' % self.stxm_filename)

        # Create the ptyd file name if not specified
        if self.info.dfile is None:
            home = Paths(IO_par).home
            self.info.dfile = '%s/prepdata/data_%d.ptyd' % (home, self.info.recipe.scan_number)
            log(3, 'Save file is %s' % self.info.dfile)

    def load_common(self):
        """
        Here we load:
            - The dark frame
            - The detector flat frame
            - The scan dimensions
        """
        common = u.Param()
        key = NXS_PATHS.frame_pattern
        if self.info.recipe.dark_number is not None:
            self.dark_nxs_filename = self.info.recipe.dark_nxs_file_pattern % self.info.recipe
            #dark = io.h5read(self.dark_nxs_filename,key)[key][0,0,:,:]# this was a problem with the dark collection. a 2x2 grid was collected.
            dark = io.h5read(self.dark_nxs_filename, key)[key]
            if dark.ndim == 4:
                dark.resize((dark.shape[0] * dark.shape[1], dark.shape[2], dark.shape[3]))
            if dark.ndim == 3:
                dark = np.median(dark, axis=0)
        else:
            dark = self.info.recipe.dark_value
        
        if self.info.recipe.detector_flat_file is not None:
            flat = io.h5read(self.info.recipe.detector_flat_file,FLAT_PATHS.key)[FLAT_PATHS.key]

        else:
            flat = 1.
        
        scan_dimensions = io.h5read(self.nxs_filename,key)[key].shape[0],io.h5read(self.nxs_filename,key)[key].shape[1]

        common.dark = dark
        common.flat = flat
        common.scan_dimensions = scan_dimensions
        return common._to_dict()
                
    def load_positions(self):
     	"""
        Load the positions and return as an (N,2) array
        """
        base_path = self.info.recipe.base_path
        mmult = u.expect2(self.info.recipe.motors_multiplier)
        keyx = STXM_PATHS.motors+str(self.info.recipe.motors[0])
        keyy=STXM_PATHS.motors+str(self.info.recipe.motors[1])
        print "file name is:%s" % self.stxm_filename
        x1 = io.h5read(self.stxm_filename,keyx)
        y1 = io.h5read(self.stxm_filename,keyy)
        x ,y = np.meshgrid(x1[keyx],y1[keyy]) # grab out the positions- they are the demand positions rather than readback and are not a list of co-ords. Meshgrid to get almost the thing we need.
        x = x.flatten()*mmult[0]# flatten the meshgrids
        y = y.flatten()*mmult[1]
        positions = np.zeros((x.shape[0],2)) # make them into a (N,2) array
        positions[:,0]=x
        positions[:,1]=y
        return positions


    def load(self, indices):
        """
        Load data frames.

        :param indices:
        :return:
        """

        raw = {}  # Container for the frames
        pos = {}  # Container for the positions. Left empty here because positions are provided by self.load_positions
        weights = {}  # Container for the weights
        key = NXS_PATHS.frame_pattern
        ix = np.zeros(len(indices))        
        iy = np.zeros(len(indices))
        
        # data from I08 comes in as a [npts_x,npts_y,framesize_x,framesize_y]
        for i in range(len(indices)):
            ix[i] = int(np.mod(indices[i], self.common.scan_dimensions[1]))  # find the remainder - works out the column
            iy[i] = int(indices[i]//self.common.scan_dimensions[0])  # works out the row
            raw[i] = (io.h5read(self.nxs_filename, key, slice=(ix[i], iy[i]))[key].astype(np.float32)-self.common.dark)/ (self.common.flat)  #-self.common.dark) # load in the data and convert type
            # TODO: update this line when the statistical weights are implemented in ptypy
            # For now, this is just a mask
            #weights[i] = (raw[i] >= 0.)
            raw[i] *= (raw[i] >= 0.)
        return raw, pos, weights

#io.h5read('somefile.h5', data[(slice1, slice2)]
