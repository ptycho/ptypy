"""
This module provides simulated 3D Bragg data. NOTE that the data here
no longer reflects what is produced at the beamline.
"""

import ptypy
from ptypy.core.data import PtyScan
import ptypy.utils as u
from ptypy import defaults_tree

from ptypy.experiment import register
from ptypy.utils.descriptor import EvalDescriptor
import h5py
import numpy as np
import time

logger = u.verbose.logger

@register()
#@defaults_tree.parse_doc('scandata.NanomaxBraggJune2017')
class NanomaxBraggJune2017(PtyScan):
    """
    Reads an early Bragg 3d ptycho format from Nanomax, multiple fly
    scans are run with rocking angle steps in between.

    Defaults:

    [name]
    default = NanomaxBraggJune2017
    type = str
    help = PtyScan subclass identifier

    [scans]
    default = []
    type = list
    help = List of scan numbers

    [theta_bragg]
    default = 0.0
    type = float
    help = Bragg angle

    [datapath]
    default = None
    type = str
    help = Path to folder containing the Sardana master file
    doc =

    [datafile]
    default = None
    type = str
    help = Sardana master file
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
        logger.info('*** load_common')
        angles = []
        for scannr in self.p.scans:
            with h5py.File(self.p.datapath + self.p.datafile) as fp:
                angles.append(float(fp.get('entry%d'%scannr + '/measurement/gonphi').value))
        logger.info(angles)
        step = np.mean(np.diff(sorted(angles)))
        logger.info(step)
        return {
            'rocking_step': step,
            'n_rocking_positions': len(angles),
            'theta_bragg': self.p.theta_bragg,
            }

    def load_positions(self):
        """
        For the 3d Bragg model, load_positions returns N-by-4 positions,
        (angle, x, z, y). The angle can be relative or absolute, the
        model doesn't care, but it does have to be uniformly spaced for
        the analysis to make any sense.

        Let's load the positions (and images) in the order they were
        acquired: x fastest, then y, then scan number in the order
        provided.
        """
        logger.info('*** load_positions')

        # first, calculate mean x and y positions for all scans, they
        # have to match anyway so may as well average them.
        x, y = [], []
        for scan in self.p.scans:
            with h5py.File(self.p.datapath + self.p.datafile) as fp:
                if scan == self.p.scans[0]:
                    # first pass: find out how many zeros to remove from the samx buffer
                    entry = 'entry%d' % scan
                    tmp = np.array(fp[entry + '/measurement/AdLinkAI_buff'])
                    for i in range(tmp.shape[1]):
                        if np.allclose(tmp[:, i:], 0.0):
                            cutoff = i
                            logger.info('using %i samx values' % cutoff)
                            break
                x.append(np.array(fp[entry + '/measurement/AdLinkAI_buff'][:, :cutoff]))
                y.append(np.array(fp[entry + '/measurement/samy']))
        x_mean = -np.mean(x, axis=0) * 1e-6
        y_mean = np.mean(y, axis=0) * 1e-6
        Nx = x_mean.shape[1]
        Ny = x_mean.shape[0]
        Nxy = Nx * Ny
        assert Ny == y_mean.shape[0]
        logger.info('Scan positions are Nx=%d, Ny=%d, Nxy=%d' % (Nx, Ny, Nxy))

        # save these numbers for the diff image loader
        self.Nx = Nx
        self.Ny = Ny
        self.Nxy = Nxy

        # then, go through the scans and fill in a N-by-4 array of positions per diff pattern
        x = np.empty(len(self.p.scans) * Nx * Ny, dtype=float)
        y = np.copy(x)
        theta = np.copy(x)
        z = np.zeros_like(x)
        for i in range(len(self.p.scans)):
            with h5py.File(self.p.datapath + self.p.datafile) as fp:
                entry = 'entry%d' % self.p.scans[i]
                th = fp[entry + '/measurement/gonphi'].value[0]
            x[i*Nxy:(i+1)*Nxy] = x_mean.flatten()
            y[i*Nxy:(i+1)*Nxy] = np.repeat(y_mean, Nx)
            theta[i*Nxy:(i+1)*Nxy] = th

        # adapt to our geometry
        tmp = z.copy()
        z = x   # our x motor goes toward positive z on the sample
        y = y   # our y motor goes toward more positive y on the sample
        x = tmp # this is the unimportant direction

        return np.vstack([theta, x, z, y]).T

    def load(self, indices):
        """
        This function returns diffraction image indexed from top left as
        viewed along the beam, i e they have (-q1, q2) indexing. PtyScan
        can always flip/rotate images.
        """
        logger.info('*** load')
        raw, positions, weights = {}, {}, {}

        for ind in indices:
            scan = self.p.scans[ind // self.Nxy] # which scan to look in
            file = (ind % self.Nxy) // self.Nx   # which line of this scan
            frame = (ind % self.Nxy) % self.Nx   # which frame of this line
            with h5py.File(self.p.datapath + self.p.detfilepattern % (scan, file)) as fp:
                data = np.array(fp['entry_0000/measurement/Merlin/data'][frame])
            raw[ind] = data

        return raw, positions, weights

    def load_weight(self):
        logger.info('*** load_weight')
        with h5py.File(self.p.maskfile) as fp:
            mask = np.array(fp['mask'])
        return mask

