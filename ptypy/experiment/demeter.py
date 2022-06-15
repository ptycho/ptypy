
""" 
    Implementation of PtyScan subclasses to hold SOLARIS-Demeter scan data.
    The beamline is developing, as is the data format.
"""
 
from ..core.data import PtyScan
from .. import utils as u
from . import register
logger = u.verbose.logger
 
import numpy as np
try:
                import hdf5plugin
except ImportError:
                logger.warning('Couldnt find hdf5plugin - better hope your h5py has bitshuffle!')
import h5py
from  PIL import Image
import os
import os.path
import time
 
 
 
@register()
class Demeter_Jun2022_simple(PtyScan):
    """
    Starting a fresh class here.
 
    Defaults:
 
    [name]
    default = Demeter_Jun2022_simple
    type = str
    help =
 
    [folder_diff]
    default = None
    type = str
    help = abosulte path to the directory containing all the diffraction patterns
    doc =
 
    [folder_dark]
    default = None
    type = str
    help = abosulte path to the directory containing all the dark fields
    doc =

    [Npoints_x]
    default = None
    type = int
    help = number of steps in x direction
    doc =
 
    [Npoints_y]
    default = None
    type = int
    help = number of steps in y direction
    doc =
 
    [fast_axis]
    default = x
    type = str
    help = which axis was the fat one
    doc =
 
    [step_size_x]
    default = 0
    type = float
    help = step size in the x direction in m
    doc =

    [step_size_y]
    default = 0
    type = float
    help = step size in the y direction in m
    doc =

    """
 
    def load_positions(self):
        """
        Provides the relative sample positions inside the scan.
        """
 
        pos_x = np.linspace(0, (self.info.Npoints_x-1)*self.info.step_size_x, self.info.Npoints_x)
        pos_y = np.linspace(0, (self.info.Npoints_y-1)*self.info.step_size_y, self.info.Npoints_y)
 
        x, y = [], []
        if self.info.fast_axis == 'x':
            for py in pos_y:
                for px in pos_x:
                    x.append(px)
                    y.append(py)
        elif self.info.fast_axis == 'y':
            for px in pos_x:
                for py in pos_y:
                    x.append(px)
                    y.append(py)
        x, y = np.array(x), np.array(y)

        positions = np.vstack((y, x)).T
        return positions
 
    def load(self, indices):
        """
        Provides the raw diffraction pattern from the detector (eiger .h5) file.
        Normalizes them, if there is normalization data.
        """
 
        raw, weights, positions = {}, {}, {}
 

        fnames_dark = os.listdir(self.info.folder_dark)
        dark_frames = np.array([np.array(Image.open(self.info.folder_dark+x)) for x in fnames_dark])
        dark = np.mean(dark_frames, axis=0)


        def find_index(fname):
            return int(fname.split('_')[-1].split('.')[0])

        fnames_diff = sorted(os.listdir(self.info.folder_diff), key=find_index)

        for ind in indices:
            raw[ind] = np.array(Image.open(self.info.folder_diff+fnames_diff[ind]))-dark
            raw[ind][raw[ind]<0] = 0
 
        return raw, positions, weights
 
    def load_weight(self):
        """
        Provides the mask used for every diffraction pattern in the whole scan
        This mask will have the shape of the first frame.
        """
 
        r, w, p = self.load(indices=(0,))
        data = r[0]
        mask = np.ones_like(data)
        
        return mask

@register()
class Demeter_Jun2022_double(PtyScan):
    """
    Starting a fresh class here.
 
    Defaults:
 
    [name]
    default = Demeter_Jun2022_double
    type = str
    help =
    doc =
 
    [folder_diff]
    default = None
    type = str
    help = abosulte path to the directory containing all the diffraction patterns
    doc =
 
    [folder_dark]
    default = None
    type = str
    help = abosulte path to the directory containing all the dark fields
    doc =

    [high_first]
    default = None
    type = int
    help = are the even or the odd frames the hogh exposure ones
    doc =

    [factor_low_to_high]
    default = None
    type = float
    help = how much brighter are the high intensity exposures
    doc =

    [high_threshhold]
    default = None
    type = float
    help = from which pixel value on are high exposure frames considered over exposed
    doc =

    [Npoints_x]
    default = None
    type = int
    help = number of steps in x direction
    doc =
 
    [Npoints_y]
    default = None
    type = int
    help = number of steps in y direction
    doc =
 
    [fast_axis]
    default = x
    type = str
    help = which axis was the fat one
    doc =
 
    [step_size_x]
    default = 0
    type = float
    help = step size in the x direction in m
    doc =

    [step_size_y]
    default = 0
    type = float
    help = step size in the y direction in m
    doc =

    """
 
    def load_positions(self):
        """
        Provides the relative sample positions inside the scan.
        """
 
        pos_x = np.linspace(0, (self.info.Npoints_x-1)*self.info.step_size_x, self.info.Npoints_x)
        pos_y = np.linspace(0, (self.info.Npoints_y-1)*self.info.step_size_y, self.info.Npoints_y)
 
        x, y = [], []
        if self.info.fast_axis == 'x':
            for py in pos_y:
                for px in pos_x:
                    x.append(px)
                    y.append(py)
        elif self.info.fast_axis == 'y':
            for px in pos_x:
                for py in pos_y:
                    x.append(px)
                    y.append(py)
        x, y = np.array(x), np.array(y)

        positions = np.vstack((y, x)).T
        return positions
 
    def load(self, indices):
        """
        Provides the raw diffraction pattern from the detector (eiger .h5) file.
        Normalizes them, if there is normalization data.
        """
 
        def find_index(fname):
            return int(fname.split('_')[-1].split('.')[0])

        def make_hdr_frame(I_low, I_high, dark_low, dark_high ,threshhold, factor):
            """
            factor ... how much more exposre the high exposure got compared to the low exposure
            """

            frame = 1.*(I_high-dark_high)/factor
            frame[I_high>=threshhold] = (I_low-dark_low)[I_high>=threshhold]
            return frame

        def load_and_average_dark(fnames):
            dark_frames = np.array([np.array(Image.open(self.info.folder_dark+x)) for x in fnames_dark_even])
            return np.mean(dark_frames, axis=0)

        raw, weights, positions = {}, {}, {}
 
        fnames_dark = np.array(os.listdir(self.info.folder_dark))
        fnames_dark_even = fnames_dark[0::2]
        fnames_dark_odd = fnames_dark[1::2]
        dark_even = load_and_average_dark(fnames_dark_even)
        dark_odd = load_and_average_dark(fnames_dark_odd)

        fnames_diff = sorted(os.listdir(self.info.folder_diff), key=find_index)
        fnames_diff_even = fnames_diff[0::2]
        fnames_diff_odd = fnames_diff[1::2]

        for ind in indices:
            frame_even = np.array(Image.open(self.info.folder_diff+fnames_diff_even[ind]))
            frame_odd = np.array(Image.open(self.info.folder_diff+fnames_diff_odd[ind]))

            if self.info.high_first >=1:
                raw[ind] = make_hdr_frame(I_low = frame_odd, 
                                          I_high = frame_even,
                                          dark_low = dark_odd, 
                                          dark_high = dark_even, 
                                          threshhold = self.info.high_threshhold, 
                                          factor = self.info.factor_low_to_high)
            else:
                raw[ind] = make_hdr_frame(I_low = frame_even, 
                                          I_high = frame_odd,
                                          dark_low = dark_even, 
                                          dark_high = dark_odd, 
                                          threshhold = self.info.high_threshhold, 
                                          factor = self.info.factor_low_to_high)       
            raw[ind][raw[ind]<0] = 0
 
        return raw, positions, weights
 
    def load_weight(self):
        """
        Provides the mask used for every diffraction pattern in the whole scan
        This mask will have the shape of the first frame.
        """
 
        r, w, p = self.load(indices=(0,))
        data = r[0]
        mask = np.ones_like(data)
        
        return mask

