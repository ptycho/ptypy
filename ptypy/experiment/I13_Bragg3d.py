"""
copied from nanomax3d

Written by Susanna Hammarberg for experiment mt20167-1

"""
import ptypy
from ptypy.core.data import PtyScan
import ptypy.utils as u
from ptypy import defaults_tree
import os

from ptypy.experiment import register 
from ptypy.utils.descriptor import EvalDescriptor
import h5py
import numpy as np
import time

logger = u.verbose.logger

@register() 
#@defaults_tree.parse_doc('scandata.NanomaxBraggJune2017')
class I13Bragg3d(PtyScan):
    """
    Reads Bragg ptycho data from I13.

    Defaults:

    [name]
    default = I13_Bragg3d
    type = str
    help = PtyScan subclass identifier

    [scans]
    default = []
    type = list
    help = List of scan numbers

    [theta_bragg]
    default = None
    type = float
    help = Bragg angle

    [datapath]
    default = None
    type = str
    help = Path to folder containing the Sardana master file
    doc =

    [maskfile]
    default = None
    type = str
    help = Arbitrary mask file
    doc = Hdf5 file containing an array called 'mask' at the root level.

    [detfilepattern]
    default = None
    type = str
    help = Format string for detector image files
    doc = A format string with two integer fields, the first holds the scan number while the second holds the image number.

    [vertical_shift]
    default = [0]
    type = list
    help = List of vertical shifts
    
    [horizontal_shift]
    default = [0]
    type = list
    help = List of horizontal shifts
    
    [detector_roi_indices]
    default = [0, 512, 0, 512]    
    type = list
    help = Indices for detector roi
 	
    [rocking_step]
    default = None
    type = float
    help = sheat since we dont have theta positions jet
	
    """

    def __init__(self, pars=None, **kwargs):
        self.p = self.DEFAULT.copy(99)
        self.p.update(pars)
        self.p.update(kwargs)
        super(self.__class__, self).__init__(self.p)

        

    def load_common(self):
        """
        We have to communicate the number of rocking positions that the
        model should expect, otherwise it never knows when there is data
        for a complete POD. We also have to specify a single number for
        the rocking step size.
        """
        print '*** load_common'

        step = np.mean(np.diff(sorted(self.thetas)))
        n_rocking = len(self.thetas)
        theta_bragg = np.median(self.thetas) / 2.0

        return {
        'rocking_step': step,
        'n_rocking_positions': n_rocking,
        'theta_bragg': theta_bragg,
        }

    def load_positions(self):
        """
        For the 3d Bragg model, load_positions returns N-by-4 positions,
        (angle, x, z, y). The angle can be relative or absolute, the
        model doesn't care, but it does have to be uniformly spaced for
        the analysis to make any sense.
        
        """
        print '*** load_positions'

        # first, calculate mean x and y positions for all scans, they
        # have to match anyway so may as well average them.
        x, y = [], []

        print 'NOTE!!! lx and ly are exchanged in the nexus I13 files as of Sept 2018. Swapping them.'
        key_pi_x = 'entry1/instrument/t1_pi_ly/t1_pi_ly'
        key_pi_y = 'entry1/instrument/t1_pi_lx/t1_pi_lx'
        for scan_nbr in self.p.scans:
            # open metadata file
            with h5py.File(os.path.join(self.p.datapath, str(scan_nbr) + '.nxs'), 'r') as fp:
                # save piezo motorpositioner            
                y.append(np.array(fp.get(key_pi_y)))
                x.append(np.array(fp.get(key_pi_x)))
        lx_mean = np.mean(x, axis=0) * 1e-6
        ly_mean = np.mean(y, axis=0) * 1e-6

        Nxy = len(lx_mean)

        # save these numbers for the diff image loader
        self.Nxy = Nxy

        # then, go through the scans and fill in a N-by-4 array of positions per diff pattern
        lx = np.tile(lx_mean, len(self.p.scans))
        ly = np.tile(ly_mean, len(self.p.scans))
        theta = np.zeros_like(lx)
        lz = np.zeros_like(lx)

        if None in (self.p.theta_bragg, self.p.rocking_step):
            print 'attempting to read rocking angles from nxs info'
            key_theta = 'entry1/before_scan/t1_theta/t1_theta'
            theta = []
            for scan_nbr in self.p.scans:
                with h5py.File(os.path.join(self.p.datapath, str(scan_nbr) + '.nxs'), 'r') as fp:
                    theta.append(fp[key_theta])
            self.thetas = theta
        else:
            print 'generating rocking angles from input parameters'
            step = self.p.rocking_step
            n_rocking = len(self.p.scans)
            theta_bragg = self.p.theta_bragg
            #TODO does only work for an uneven number of angles
            
            self.thetas = np.linspace(theta_bragg-step*n_rocking/2.0, theta_bragg+step*n_rocking/2.0, n_rocking)

        for i, th in enumerate(self.thetas):
            theta[i*Nxy:(i+1)*Nxy] = th

        # adapt to our geometry
        # ...so in the Berenguer frame, we have these coordinates (beam position on sample):
        z = -lx*np.cos(np.deg2rad(self.p.theta_bragg))
        y = -ly
        x = lz # this is the unimportant direction
        
        return np.vstack([theta, x, z, y]).T

    def load(self, indices):
        """
        This function returns diffraction image indexed from top left as
        viewed along the beam, i e they have (-q1, q2) indexing. PtyScan
        can always flip/rotate images.

        """
        print '*** load'
        raw, positions, weights = {}, {}, {}
        
        detind = self.p.detector_roi_indices

        for ind in indices:
            scan = self.p.scans[ind // self.Nxy] # which scan to look in
            frame = ind % self.Nxy               # which frame in this scan
            with h5py.File(os.path.join(self.p.datapath, str(scan) + '.nxs'), 'r') as fp:
                dat = fp['entry1/instrument/excalibur/data'][frame, detind[0]:detind[1], detind[2]:detind[3]]
            raw[ind] = dat

        return raw, positions, weights

    def load_weight(self):
        print '*** load_weight'
        detind = self.p.detector_roi_indices
        with h5py.File('/dls_sw/i13-1/scripts/Darren/detector_masks/82245.nxs', 'r') as fp:
            weights = fp['entry1/excalibur_config_normal/data'][0, detind[0]:detind[1], detind[2]:detind[3]]
        return weights

