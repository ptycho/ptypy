"""   
This module provides simulated 3D Bragg data. 
"""

import ptypy
from ptypy.core.data import PtyScan
import ptypy.utils as u
from ptypy import defaults_tree
from ptypy.experiment import register
from ptypy.utils.descriptor import EvalDescriptor
from ptypy.core import geometry_bragg
from ptypy.core import illumination
from ptypy.core import xy

import numpy as np
import time

logger = u.verbose.logger

@register()
class Bragg3dSimScan(PtyScan):
    """
    Provides simulated 3D Bragg data based on the numerical 
    experiment in Berenguer et al., PRB 88 (2013) 144101. The settings
    from that paper are very expensive, so the defaults here are simpler
    and do not necessarily make physical sense.

    Defaults:

    [name]
    default = Bragg3dSimScan
    type = str
    help = PtyScan subclass identifier

    [shape]
    default = 256

    [distance]
    default = 2

    [psize]
    default = 13e-6

    [energy]
    default = 8.5

    [rocking_step]
    default = .0025
    type = float
    help = Step size in the rocking curve in degrees

    [n_rocking_positions]
    default = 40
    type = int
    help = Number of rocking positions

    [theta_bragg]
    default = 22.32
    type = float
    help = Bragg angle in degrees

    [shuffle]
    default = False
    type = bool
    help = Shuffles all diffraction patterns
    doc = Mainly to test that they are still assembled correctly.

    [dump]
    default = None
    type = str
    help = Dump raw simulated 3d diffraction data to npz file

    [dry_run]
    default = False
    type = bool
    help = Don't calculate diffraction patterns
    doc = Skips the heavy FFT and just returns empty diff patterns.

    [illumination.aperture.form]
    default = 'rect'
    type = str
    help = default override, see :any:`scan.Full.illumination`
    doc =

    [illumination.aperture.size]
    default = 3e-6
    type = float, tuple
    help = default override, see :any:`scan.Full.illumination`
    doc =

    [scantype]
    default = '1d'
    type = str
    help = Type of position scan
    doc = '1d' for scan along y as in the paper, '2d' for xy spiral scan

    [stepsize]
    default = .5e-6
    type = float
    help = Step size of the spiral scan

    """

    def __init__(self, pars=None, **kwargs):
        self.p = self.DEFAULT.copy(99)
        self.p.update(pars)
        self.p.update(kwargs)
        super(Bragg3dSimScan, self).__init__(self.p)

        # do the simulation
        self.simulate()

    def simulate(self):
        ### Set up a 3D geometry
        shape = tuple(u.expect2(self.p.shape))
        psize = tuple(u.expect2(self.p.psize))
        g = ptypy.core.geometry_bragg.Geo_Bragg(
            psize=(self.p.rocking_step,) + psize, 
            shape=(self.p.n_rocking_positions,) + shape,
            energy=self.p.energy, 
            distance=self.p.distance, 
            theta_bragg=self.p.theta_bragg)

        logger.info('Data will be simulated with these geometric parameters:')
        logger.info(g)

        ### Set up scan positions in the xy plane
        if self.p.scantype == '2d':
            pos = u.Param()
            pos.spacing = self.p.stepsize
            pos.extent = (2e-6, 5e-6)
            pos.model = 'spiral'
            pos = xy.from_pars(pos)
            Npos = pos.shape[0]
            positions = np.zeros((Npos, 3))
            positions[:, 0] = pos[:, 0]
            positions[:, 2] = pos[:, 1]
        elif self.p.scantype == '1d':
            pos = np.arange(-2.5e-6, 2.5e-6, self.p.stepsize)
            Npos = len(pos)
            positions = np.zeros((Npos, 3))
            positions[:, 2] = pos

        ### Set up the object and its views

        # Create a container for the object array, which will represent
        # the object in the non-orthogonal coordinate system conjugate
        # to the q-space measurement frame.
        C = ptypy.core.Container(data_type=np.complex128, data_dims=3)

        # For each scan position in the orthogonal coordinate system,
        # find the natural coordinates and create a View instance there.
        views = []
        for pos in positions:
            pos_ = g._r3r1r2(pos)
            views.append(ptypy.core.View(C, storageID='Sobj', 
                psize=g.resolution, coord=pos_, shape=g.shape))
        S = C.storages['Sobj']
        C.reformat()

        # Define the test sample based on the orthogonal position of
        # each voxel. First, the cartesian grid is obtained from the
        # geometry object, then this grid is used as a condition for the
        # sample's magnitude.
        xx, zz, yy = g.transformed_grid(S, input_space='real',
            input_system='natural')
        S.fill(0.0)
        S.data[(zz >= -90e-9) & (zz < 90e-9) & (yy + .3*zz >= 1e-6) &
            (yy - .3*zz< 2e-6) & (xx < 1e-6)] = 1
        S.data[(zz >= -90e-9) & (zz < 90e-9) & (yy + .3*zz >= -2e-6) &
            (yy - .3*zz < -1e-6)] = 1

        # save this for possible export
        self.simulated_object = S

        ### Set up the probe and calculate diffraction patterns

        # First set up a two-dimensional representation of the incoming
        # probe, with arbitrary pixel spacing.
        extent = max(g.probe_extent_vs_fov())
        psize = g.resolution.min() / 5
        shape = int(np.ceil(extent / psize))
        logger.info('Generating incoming probe %d x %d (%.3e x %.3e) with psize %.3e...'
            % (shape, shape, extent, extent, psize))
        t0 = time.time()

        Cprobe = ptypy.core.Container(data_dims=2, data_type='float')
        Sprobe = Cprobe.new_storage(psize=psize, shape=shape)

        # fill the incoming probe
        illumination.init_storage(Sprobe, self.p.illumination, energy=g.energy)
        logger.info('...done in %.3f seconds' % (time.time() - t0))

        # The Bragg geometry has a method to prepare a 3d Storage by extruding
        # the 2d probe and interpolating to the right grid. The returned storage
        # contains a single view compatible with the object views.
        Sprobe_3d = g.prepare_3d_probe(Sprobe, system='natural')
        probeView = Sprobe_3d.views[0]

        # Calculate diffraction patterns by using the geometry's propagator.
        diff = []
        if self.p.dry_run:
            for v in views:
                diff.append(np.zeros(probeView.shape))
        else:
            for v in views:
                diff.append(np.abs(g.propagator.fw(
                    v.data * probeView.data))**2)

        # dump the 3d arrays for testing
        if self.p.dump is not None:
            np.savez(self.p.dump, 
                **{'diff%02d'%i : diff[i] for i in range(len(diff))})

        # stack the 2d diffraction patterns and save
        self.diff = []
        for i in range(len(diff)):
            for j in range(len(diff[i])):
                self.diff.append(diff[i][j,:,:])

        # convert the positions from (x, z, y) to (angle, x, z, y) and
        # save, we need the angle and in future we won't know in which
        # plane the scan was done (although here it is in xy). these xyz
        # axis still follow Berenguer et al PRB 2013.
        self.positions = np.empty((g.shape[0] * Npos, 4), dtype=float)
        angles = (np.arange(g.shape[0]) - g.shape[0] / 2.0 + 1.0/2) * g.psize[0]
        for i in range(Npos):
            for j in range(g.shape[0]):
                self.positions[i * g.shape[0] + j, 1:] = positions[i, :]
                self.positions[i * g.shape[0] + j, 0] = angles[j]

        # shuffle everything as a test
        if self.p.shuffle:
            order = list(range(len(self.diff)))
            from random import shuffle
            shuffle(order)
            self.diff = [self.diff[i] for i in order]
            new_pos = np.empty_like(self.positions)
            for i in range(len(new_pos)):
                new_pos[i] = self.positions[order[i]]
            self.positions = new_pos

    def load_common(self):
        """
        We have to communicate the number of rocking positions that the
        model should expect, otherwise it never knows when there is data
        for a complete POD.
        """
        return {
            'rocking_step': self.p.rocking_step,
            'n_rocking_positions': self.p.n_rocking_positions,
            'theta_bragg': self.p.theta_bragg,
            }

    def load_positions(self):
        """
        For the 3d Bragg model, load_positions returns N-by-4 positions,
        (angle, x, z, y). The angle can be relative or absolute, the
        model doesn't care, but it does have to be uniformly spaced for
        the analysis to make any sense.
        """
        return self.positions

    def load(self, indices):
        """
        This function returns diffraction image indexed from top left as
        viewed along the beam, i e they have (-q1, q2) indexing. PtyScan
        can always flip/rotate images.
        """
        raw, positions, weights = {}, {}, {}

        # pick out the requested indices
        for i in indices:
            raw[i] = self.diff[i][::-1,:]

        return raw, positions, weights

    def load_weight(self):
        return np.ones_like(self.diff[0])

defaults_tree['scandata.Bragg3dSimScan'].add_child(illumination.illumination_desc, copy=True)
defaults_tree['scandata.Bragg3dSimScan.illumination'].prune_child('diversity')


if __name__ == '__main__':
    u.verbose.set_level(3)
    ps = Bragg3dSimScan()
    ps.initialize()
    while True:
        msg = ps.auto(23)
        if msg == ps.EOS:
            break
        logger.info('Got %d images' % len(msg['iterable']))
