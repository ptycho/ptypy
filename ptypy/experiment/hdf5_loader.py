# -*- coding: utf-8 -*-
"""\
Scan loading recipe for the I13 beamline, Diamond.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import h5py as h5
import numpy as np

from ptypy import io
from ptypy import utils as u
from ptypy.core import Ptycho
from ptypy.core.data import PtyScan
from ptypy.core.paths import Paths
from ptypy.utils.descriptor import defaults_tree
from ptypy.utils.verbose import log
from ptypy.utils import parallel

IO_par = Ptycho.DEFAULT['io']

@defaults_tree.parse_doc('scandata.Hdf5Loader')
class Hdf5Loader(PtyScan):
    """
    First attempt to make a generalised hdf5 loader for data. Please raise a ticket in github if changes are required
    so we can coordinate. There will be a Nexus and CXI subclass to this in the future.

    Defaults:

    [name]
    default = 'Hdf5'
    type = str
    help =

    [intensities]
    default =
    type = Param
    help = This parameter contains the diffraction data. Data shapes can be either (A, B, frame_size_m, frame_size_n) or
            (C, frame_size_m, frame_size_n). It's assumed in this latter case that the fast axis in the scan corresponds
            the fast axis on disc (i.e. C-ordered layout).

    [intensities.is_swmr]
    default = False
    type = bool
    help = If this is set to be true, then intensities are assumed to be a swmr dataset that is being written as processing
            is taking place.

    [intensities.live_key]
    default = None
    type = str
    help = If intensities.is_swmr is true then we need a live_key to know where the data collection has progressed to.
            This is the key to these live keys inside the intensities.file. They are zero at the scan start, but non-zero
            when the position is complete.

    [intensities.file]
    default = None
    type = str
    help = This is the path to the file containing the diffraction intensities.

    [intensities.key]
    default = None
    type = str
    help = This is the key to the intensities entry in the hdf5 file.

    [positions]
    default =
    type = Param
    help = This parameter contains the position information data. Shapes for each axis that are currently covered and
            tested corresponding to the intensity shapes are:
                                 axis_data.shape (A, B) for data.shape (A, B, frame_size_m, frame_size_n),
                                 axis_data.shape (k,) for data.shape (k, frame_size_m, frame_size_n),
                                 axis_data.shape (C, D) for data.shape (C*D, frame_size_m, frame_size_n) ,
                                 axis_data.shape (C,) for data.shape (C, D, frame_size_m, frame_size_n) where D is the
                                 size of the other axis,
                            and  axis_data.shape (C,) for data.shape (C*D, frame_size_m, frame_size_n) where D is the
                            size of the other axis.

    [positions.is_swmr]
    default = False
    type = bool
    help = If this is set to be true, then positions are assumed to be swmr datasets that are being written as processing
            is taking place.

    [positions.live_key]
    default = None
    type = str
    help = If positions.is_swmr is true then we need a live_key to know where the data collection has progressed to.
            This is the key to these live keys inside the positions.file. If None, whilst positions.is_swmr is True,
            then we just assume the same keys work for both positions and intensities. They are zero at the scan start,
            but non-zero when the position is complete.

    [positions.file]
    default = None
    type = str
    help = This is the path to the file containing the  position information. If None then we try to find the information
            in the "intensities.file" location.

    [positions.slow_key]
    default = None
    type = str
    help = This is the key to the slow-axis positions entry in the hdf5 file.

    [positions.slow_multiplier]
    default = 1.0
    type = float
    help = This is a scaling factor to get the motor position into metres.

    [positions.fast_key]
    default = None
    type = str
    help = This is the key to the fast-axis positions entry in the hdf5 file.

    [positions.fast_multiplier]
    default = 1.0
    type = float
    help = This is a scaling factor to get the motor position into metres.

    [mask]
    default =
    type = Param
    help = This parameter contains the mask data. The  shape is assumed to be (frame_size_m, frame_size_n) or the same
            shape of the full intensities data.

    [mask.file]
    default = None
    type = str
    help = This is the path to the file containing the diffraction mask.

    [mask.key]
    default = None
    type = str
    help = This is the key to the mask entry in the hdf5 file.

    [flatfield]
    default =
    type = Param
    help = This parameter contains the flatfield data. The shape is assumed to be (frame_size_m, frame_size_n) or the same
            shape of the full intensities data.

    [flatfield.file]
    default = None
    type = str
    help = This is the path to the file containing the diffraction flatfield.

    [flatfield.key]
    default = None
    type = str
    help = This is the key to the flatfield entry in the hdf5 file.

    [darkfield]
    default =
    type = Param
    help = This parameter contains the darkfield data.The shape is assumed to be (frame_size_m, frame_size_n) or the same
            shape of the full intensities data.

    [darkfield.file]
    default = None
    type = str
    help = This is the path to the file containing the diffraction darkfield.

    [darkfield.key]
    default = None
    type = str
    help = This is the key to the darkfield entry in the hdf5 file.

    [normalisation]
    default =
    type = Param
    help = This parameter contains information about the per-point normalisation (i.e. ion chamber reading).
            It is assumed to have the same dimensionality as data.shape[:-2]

    [normalisation.is_swmr]
    default = False
    type = bool
    help = If this is set to be true, then normalisations are assumed to be swmr datasets that are being written as processing
            is taking place.

    [normalisation.live_key]
    default = None
    type = str
    help = If normalisation.is_swmr is true then we need a live_key to know where the data collection has progressed to.
            This is the key to these live keys inside the normalisation.file. If None, whilst normalisation.is_swmr is
            True, then we just assume the same keys work for both normalisation and intensities. They are zero at the
            scan start, but non-zero when the position is complete.

    [normalisation.file]
    default = None
    type = str
    help = This is the path to the file containing the normalisation information. If None then we try to find the information
            in the "intensities.file" location.

    [normalisation.key]
    default = None
    type = str
    help = This is the key to the normalisation entry in the hdf5 file.

    [recorded_energy]
    default =
    type = Param
    help = This parameter contains information if we are use the recorded energy rather than as a parameter.
            It should be a scalar value.
    
    [recorded_energy.file]
    default = None
    type = str
    help = This is the path to the file containing the recorded_energy.

    [recorded_energy.key]
    default = None
    type = str
    help = This is the key to the recorded_energy entry in the hdf5 file.

    [recorded_distance]
    default =
    type = Param
    help = This parameter contains information if we are use the recorded distance to the detector rather than as a parameter,
            It should be a scalar value.
    
    [recorded_distance.file]
    default = None
    type = str
    help = This is the path to the file containing the recorded_distance between sample and detector.

    [recorded_distance.key]
    default = None
    type = str
    help = This is the key to the recorded_distance entry in the hdf5 file.

    """

    def __init__(self, pars=None, **kwargs):
        """
        I13 (Diamond Light Source) data preparation class.
        """
        self.p = self.DEFAULT.copy(99)
        self.p.update(pars)
        super(Hdf5Loader, self).__init__(self.p, **kwargs)
        self._scantype = None
        self._ismapped = None
        self.intensities = None
        self.slow_axis = None
        self.fast_axis = None
        # lets raise some exceptions here for the essentials
        if None in [self.p.intensities.file,
                    self.p.intensities.key,
                    self.p.positions.file,
                    self.p.positions.slow_key,
                    self.p.positions.fast_key]:
            raise RuntimeError("Missing some information about either the positions or the intensity mapping!")

        log(4, u.verbose.report(self.info))

        self.intensities = h5.File(self.p.intensities.file, 'r')[self.p.intensities.key]
        data_shape = self.intensities.shape
        self.slow_axis = h5.File(self.p.positions.file, 'r')[self.p.positions.slow_key]
        positions_slow_shape = self.slow_axis.shape
        self.fast_axis = h5.File(self.p.positions.file, 'r')[self.p.positions.fast_key]
        positions_fast_shape = self.fast_axis.shape
        log(3, "The shape of the \n\tdiffraction intensities is: {}\n\tslow axis data:{}\n\tfast axis data:{}".format(data_shape,
                                                                                                                              positions_slow_shape,
                                                                                                                      positions_fast_shape))

        if data_shape[:-2] == positions_slow_shape == positions_fast_shape:
            log(3, "Everything is wonderful, each diffraction point has a co-ordinate.")
            self.num_frames = np.prod(positions_fast_shape)
            self._ismapped = True
            if len(data_shape) == 4:
                self._scantype = "raster"
            else:
                self._scantype = "arb"
        else:
            self._ismapped = False

    def load_weight(self):
        # Load mask as weight
        if (self.p.mask.key is not None) and (self.p.mask.file is not None):
            return h5.File(self.p.mask.file)[self.p.mask_key].astype(float)
        else:
            log(4, "No mask was loaded. mask.key was {} and mask.file was {}".format(self.p.mask.file, self.p.mask.key))

    def check(self, frames, start):
        """
        Returns the number of frames available from starting index `start`, and whether the end of the scan
        was reached.

        :param frames: Number of frames to load
        :param start: starting point
        :return: (frames_available, end_of_scan)
        - the number of frames available from a starting point `start`
        - bool if the end of scan was reached (None if this routine doesn't know)
        """

        return self.num_frames, True

    def load(self, indices):
        """
        Load frames given by the indices.

        :param indices:
        :return:
        """
        intensities = {}
        positions = {}
        weights = {}
        if (self._ismapped and (self._scantype is 'raster')):
            sh = self.slow_axis.shape
            for jj in indices:
                intensities[jj] = self.intensities[jj % sh[0], jj // sh[1]]  # or the other way round???
                positions[jj] = self.slow_axis[jj % sh[0], jj // sh[1]], self.fast_axis[jj % sh[0], jj // sh[1]]

        if (self._ismapped and (self._scantype is 'arb')):
            for jj in indices:
                intensities[jj] = self.intensities[jj]  # or the other way round???
                positions[jj] = self.slow_axis[jj], self.fast_axis[jj]

        log(3, 'Data loaded successfully.')
        return intensities, positions, weights



