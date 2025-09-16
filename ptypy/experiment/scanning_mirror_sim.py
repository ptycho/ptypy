import numpy as np

from ptypy.utils.verbose import logger
from ptypy import utils as u
from ptypy.core.data import MoonFlowerScan
from ptypy.core import geometry_scanning_mirror
from ptypy.experiment import register


__all__ = ['ScanningMirrorMoonFlower']


@register()
class ScanningMirrorMoonFlower(MoonFlowerScan):
    """
    As MoonFlowerScan, but using the scanning mirror propagators.

    Defaults:

    [shift_pos_factor]
    default = 0.0
    type = float, int
    help =

    """

    def __init__(self, pars=None, **kwargs):
        """
        Parent pars are for the
        """
        super(ScanningMirrorMoonFlower, self).__init__(pars=pars, **kwargs)
        self.beam_shifts = self.pos * self.p.shift_pos_factor


    def load(self, indices):
        p = self.pixel
        s = self.geo.shape
        raw = {}

        if self.p.add_poisson_noise:
            logger.info("Generating data with poisson noise.")
        else:
            logger.info("Generating data without poisson noise.")

        for k in indices:
            geo_pars = self.meta
            geo_pars.beam_shift = [list(self.beam_shifts[k, :])]
            geo = geometry_scanning_mirror.Geo_ScanningMirror(pars=geo_pars)

            intensity_j = u.abs2(self.geo.propagator.fw(
                self.pr * self.obj[p[k][0]:p[k][0] + s[0],
                                   p[k][1]:p[k][1] + s[1]]
            ))

            if self.p.psf > 0.:
                intensity_j = u.gf(intensity_j, self.p.psf)

            if self.p.add_poisson_noise:
                raw[k] = np.random.poisson(intensity_j).astype(np.int32)
            else:
                raw[k] = intensity_j.astype(np.int32)

        return raw, {}, {}

    def load_positions(self):
        positions = super(ScanningMirrorMoonFlower, self).load_positions()
        positions = np.hstack([
            positions,
            self.beam_shifts
        ])
        return positions