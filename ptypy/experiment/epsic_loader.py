# -*- coding: utf-8 -*-
"""\
Scan loading recipe for the Diamond beamlines.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import h5py as h5
import numpy as np

from ptypy.utils.verbose import log
from ptypy.experiment import register
from .hdf5_loader import Hdf5Loader, Hdf5LoaderFast


class _Epsic:
    """
    Defaults:

    [rotation]
    default = 0
    type = float
    help = Rotation of the scanning coordinate system 
           around the optical access, given in degrees

    [stepsize]
    default = 0
    type = float
    help = Step size of raster scan, given in meters

    [numpos]
    default = 256
    type = int
    help = Nr. of positions in raster scan along each direction
    """
    def _params_check(self):
        """
        raise error if essential parameters mising
        """
        if None in [self.p.intensities.file,
                    self.p.intensities.key,
                    self.p.rotation,
                    self.p.stepsize,
                    self.p.numpos]:
            raise RuntimeError("Missing some information about either the positions (rotation, stepsize, numpos) or the intensity mapping!")

    def _prepare_intensity_and_positions(self):
        """
        Prep for loading intensity and position data
        """
        self.fhandle_intensities = h5.File(self.p.intensities.file, 'r')
        self.intensities = self.fhandle_intensities[self.p.intensities.key]
        self.intensities_dtype = self.intensities.dtype
        self.data_shape = self.intensities.shape
        if self._is_spectro_scan and self.p.outer_index is not None:
            self.data_shape = tuple(np.array(self.data_shape)[1:])

        pos = self._calculate_scan_positions()
        self.fast_axis = pos[1]
        self.positions_fast_shape = self.fast_axis.shape
        self.slow_axis = pos[0]
        self.positions_slow_shape = self.slow_axis.shape

        log(3, "The shape of the \n\tdiffraction intensities is: {}\n\tslow axis data:{}\n\tfast axis data:{}".format(self.data_shape,
                                                                                                                      self.positions_slow_shape,
                                                                                                                      self.positions_fast_shape))
        if self.p.positions.skip > 1:
            log(3, "Skipping every {:d} positions".format(self.p.positions.skip))   


    def _calculate_scan_positions(self):
        """
        Calculate theoretical scan possitons for 
        scanning at ePSIC instrument.

        Returns
        -------
        pos : ndarray
            array of shape (2,N1,N2) where N1,N2 are the dimensions 
            along the slow and fast axis, respectively 
        """
        step = self.p.stepsize
        N = self.p.numpos
        beta = self.p.rotation
        xx,yy = np.meshgrid((np.arange(N)-N//2)*step, (np.arange(N)-N//2)*step)
        rot = beta * np.pi / 180.
        R0 = np.matrix([[np.cos(rot), -1 * np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        pos = -np.einsum('ji, mni -> jmn', R0, np.dstack([yy, xx]))
        return pos


@register()
class EpsicHdf5Loader(_Epsic, Hdf5Loader):
    pass

@register()
class EpsicHdf5LoaderFast(_Epsic, Hdf5LoaderFast):
    pass
