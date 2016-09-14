"""  Implementation of a PtyScan subclass to hold nanomax scan data. The
format is very preliminary, and therefore the class NanomaxTmpScan  is subject
to frequent change.  """

import ptypy
from ptypy.core.data import PtyScan
from ptypy.utils import Param
import h5py
import numpy as np

# the PtyScan class takes a parameter tree scans.xxx.data as  its constructor
# argument. These are default values of that tree, which include the default
# recipe.

# All "additional" parameters must come from the recipe tree, and the recipe
# is filled in from the script where data preparation is initiated. Here's the
# default.
RECIPE = Param()
RECIPE.dataPath = None
RECIPE.datafile = None
RECIPE.maskfile = None
RECIPE.maxFrames = None		# read only the first N frames
RECIPE.stepsize = None  	# step size in m

# These are the paths within hdf5 files
NEXUS_DATA_PATH = 'entry/detector/data'

class NanomaxTmpScan(PtyScan):

	def __init__(self, pars=None, **kwargs):

		# Get the default parameter tree and add a recipe branch. Here, the
		# parameter tree is that corresponding to a scan.data tree, I guess it
		# will eventually become a scans.xxx.data tree.
		p = PtyScan.DEFAULT.copy()
		p.recipe = RECIPE.copy()
		p.update(pars, in_place_depth=10)
		p.update(kwargs)

		# Call the base class constructor with the new updated parameters.
		# This constructor only reads params, it doesn't modify them. We've
		# already put the kwargs in p, so we don't need to pass them here.
		super(NanomaxTmpScan, self).__init__(p)

		# From now on, all the input parameters are available through self.info.
		# Some stuff is available also through self.meta.

		# should probably instantiate Geo here somewhere and get distance,
		# psize, shape from somewhere.

	def load_positions(self):  
	    # This is based on a very early hdf5 format and assumes that the scan
	    # is sqrt(Nframes)-by- sqrt(Nframes).

	    # load_positions causes self.num_frames to get set, so here we can
	    # limit how many files to read I think.

	    # The docs say that positions become available as self.positions, but
	    # instead it's in self.info.positions_scan.

		filename = self.info.recipe.dataPath + self.info.recipe.datafile
		with h5py.File(filename, 'r') as hf:
		    data = hf.get(NEXUS_DATA_PATH)
		    data = np.array(data)
		    self.nPositions = data.shape[0]
		positions = []
		for motor_y in range(int(np.sqrt(self.nPositions))):
			# tried to adapt this to Bjoern's info here
			# https://github.com/ptycho/ptypy-dev/issues/39
			for motor_x in range(int(np.sqrt(self.nPositions))):
				positions.append(np.array([-motor_y, -motor_x]) * self.info.recipe.stepsize)
				if len(positions) == self.info.recipe.maxFrames:
					return np.array(positions)

	def load(self, indices):
		# returns three dicts: raw, positions, weights, whose keys are the
		# scan pont indices. If one weight (mask) is to be used for the whole
		# scan, it should be loaded with load_weights(). The same goes for the
		# positions. We don't really need a mask here, but it must be
		# provided, otherwise it's given the shape of self.info.shape, and
		# then there's a shape mismatch in some multiplication.

		# Probably this should slice up the big array on reading, but won't bother now. 
		
		raw, weights, positions = {}, {}, {}

		filename = self.info.recipe.dataPath + self.info.recipe.datafile
		with h5py.File(filename) as hf:
			data = hf.get(NEXUS_DATA_PATH)
			for i in indices:
				raw[i] = np.asarray(data[i])
				weights[i] = np.ones(data[i].shape)

		return raw, positions, weights
