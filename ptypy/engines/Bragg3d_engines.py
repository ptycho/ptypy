# -*- coding: utf-8 -*-
"""
This module contains experimental DM engines for 3d Bragg ptycho.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
from .projectional import DM
from . import register
from ..core.manager import Bragg3dModel
from ..utils import parallel
from ..utils.verbose import logger
import time
import numpy as np

__all__ = ['DM_3dBragg']

@register()
class DM_3dBragg(DM):
    """
    DM engine adapted to 3d Bragg reconstruction. Specifically, sample
    supports are added (other regulizers might get added later). The
    engine is highly experimental and not based on any published
    algorithms, but on the notion that DM works well for 3d Bragg data
    except that the positioning along the beam path is ambiguous.

    Defaults:

    [name]
    default = DM_3dBragg
    type = str
    help =
    doc =

    [sample_support]
    default =
    type = Param
    help = Sample support settings
    doc =

    [sample_support.type]
    default = 'thinlayer'
    type = str
    help = Sample support geometry
    doc = Options are 'thinlayer' for one-dimensional support as
          function of z, 'rod' for one-dimensional radial support around
          the z axis.

    [sample_support.size]
    default = 200e-9
    type = float
    help = Support thickness or radius
    doc = This parameter is ignored when shrink wrapping is used.

    [sample_support.coefficient]
    default = 0.1
    type = float
    help = Scaling of region outside the support
    lowlim = 0.0
    uplim = 1.0
    doc = Sample amplitude is multiplied by this value outside the support region

    [sample_support.shrinkwrap]
    default =
    type = Param
    help = Shrink wrap settings. None for no shrink wrap.
    doc =

    [sample_support.shrinkwrap.smooth]
    default = 1.0
    type = float
    lowlim = .3
    help = Shrink wrap smoothing parameter in pixels
    doc = Sigma of gaussian with which to smooth object profile before applying shrink wrap. Pass None for no smoothing. Values < .3 make little sense.

    [sample_support.shrinkwrap.cutoff]
    default = .5
    type = float
    help = Shrink wrap cutoff parameter
    doc = The support is truncated where the object profile has decayed to this value relative to the maximum.

    [sample_support.shrinkwrap.monotonic]
    default = True
    type = bool
    help = Require the object profile to be monotonic
    doc = If the object profile increases again after the maximum, then the support is cut off. Set the cutoff parameter low to make this the dominating criterion.

    [sample_support.shrinkwrap.start]
    default = 10
    type = int
    help = Start shrink wrap after this iteration
    doc =

    [sample_support.shrinkwrap.plot]
    default = False
    type = bool
    help = Pass shrink wrap information to the plot client
    doc = Puts shrink wrap information in the runtime dict. The plot client can choose to plot it if it likes.

    """

    SUPPORTED_MODELS = [Bragg3dModel, ]

    def __init__(self, ptycho_parent, pars):
        """
        Need to override here, to copy the grouped "sample_support."
        parameters properly.
        """
        p = self.DEFAULT.copy(99)
        if pars is not None:
            p.update(pars, in_place_depth=99)

        super(DM_3dBragg, self).__init__(ptycho_parent, p)

    def object_update(self):
        """
        DM object update, modified with sample support. We work with a
        generalized coordinate "s", along which we calculate sample
        density profiles and apply cutoffs. The support type switches
        how this coordinate is calculated and used for cutoff. More
        types can easily be added.
        """
        super(DM_3dBragg, self).object_update()
        
        # no support
        if self.p.sample_support is None:
            return

        # access object storage and geometry through any active pod
        for name, pod in self.pods.items():
            if pod.active:
                break
        geo = pod.geometry
        S = pod.ob_view.storage
        layer = pod.ob_view.layer

        # fixed support
        if not self.p.sample_support.shrinkwrap:
            shigh = self.p.sample_support.size / 2.0
            slow = -shigh

        # shrink wrap
        elif self.curiter >= self.p.sample_support.shrinkwrap.start:
            logger.info('Shrink wrapping...')
            t0 = time.time()

            # transform to cartesian (r3, r1, r2) -> (x, z, y)
            Scart = geo.coordinate_shift(S, input_space='real',
                         input_system='natural', keep_dims=True,
                         layer=layer)
            x, z, y = Scart.grids()

            # here we calculate the object profile in the coordinate of interest
            if self.p.sample_support.type == 'thinlayer':
                s = z[layer][0, :, 0]
                sprofile = np.mean(np.abs(Scart.data[layer]), axis=(0,2))
                icenter = np.argmax(sprofile)
            if self.p.sample_support.type == 'rod':
                arr = np.sum(np.abs(Scart.data[layer]), axis=1)
                ijcenter = np.unravel_index(np.argmax(arr), arr.shape)
                xcenter = x[layer][ijcenter[0], 0, 0]
                ycenter = y[layer][0, 0, ijcenter[1]]
                # radial integration
                x_, y_ = x[layer][:, 0, :], y[layer][:, 0, :]
                r = np.sqrt((x_ - xcenter)**2 + (y_ - ycenter)**2)
                scaling = np.min(geo.resolution)
                r /= scaling
                r = r.astype(np.int)
                tbin = np.bincount(r.ravel(), arr.ravel())
                nr = np.bincount(r.ravel())
                s = np.arange(len(tbin)) * scaling
                sprofile = tbin / nr
                icenter = 0

            # gaussian smooth
            sigma = self.p.sample_support.shrinkwrap.smooth
            if self.p.sample_support.shrinkwrap.smooth:
                if sigma > 0:
                    n = 3 * sigma
                    nn = (np.arange(2 * n + 1) - n)
                    ssmooth = 1.0 / (sigma * np.sqrt(np.pi * 2)) * np.exp(-(nn/2.0/sigma)**2)
                    sprofile = np.convolve(sprofile, ssmooth, mode='same')

            # walk around from the maximum to determine where to cut off
            # the support, in each direction find the first values slow
            # and shigh that lie outside the support.
            cutoff = self.p.sample_support.shrinkwrap.cutoff
            # walk negative
            slow = s[0]
            for i in range(1, icenter):
                if (sprofile[icenter-i] / sprofile[icenter] < cutoff
                    or (self.p.sample_support.shrinkwrap.monotonic and
                        sprofile[icenter-i] > sprofile[icenter-i+1])):
                    slow = s[icenter-i]
                    break

            # walk positive
            shigh = s[len(sprofile) - 1]
            for i in range(1, len(sprofile)-icenter-1):
                if parallel.master:
                    print(sprofile[icenter+i] / sprofile[icenter])
                if (sprofile[icenter+i] / sprofile[icenter] < cutoff
                    or (self.p.sample_support.shrinkwrap.monotonic and
                        sprofile[icenter+i] > sprofile[icenter+i-1])):
                    shigh = s[icenter+i]
                    break

            # save info for runtime dict
            self.sx = s
            self.sprofile = sprofile
            self.slow = slow
            self.shigh = shigh
            logger.info('Cutting of the support coordinate at %.1e, %.1e' % (slow, shigh))
            logger.info('...done in %.2f seconds.' % (time.time() - t0))

        # shrink wrap, but not yet
        else:
            return

        # apply the support according to the coordinate of interest
        r3, r1, r2 = S.grids()
        x, z, y = pod.geometry.transformed_grid(
            (r3[layer], r1[layer], r2[layer]),
            input_space='real', input_system='natural')
        if self.p.sample_support.type == 'thinlayer':
            s = z
        elif self.p.sample_support.type == 'rod':
            try:
                s = np.sqrt((x-xcenter)**2 + (y-ycenter)**2)
            except:
                s = np.sqrt(x**2 + y**2)

        S.data[layer][(s > shigh) | (s < slow)] \
            *= self.p.sample_support.coefficient

    def _fill_runtime(self):
        """
        Hack to get shrinkwrap info into the runtime dict.
        """
        super(DM_3dBragg, self)._fill_runtime()

        try:
            assert self.p.sample_support.shrinkwrap.plot
        except (AttributeError, AssertionError):
            return

        try:
            self.ptycho.runtime.iter_info[-1]['shrinkwrap'] = [self.sx, self.sprofile, self.slow, self.shigh]
        except:
            pass
