# -*- coding: utf-8 -*-
"""\
Scan loading recipe for the I13 beamline, Diamond.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import h5py as h5
import numpy as np

from ptypy import utils as u
from ptypy.core.data import PtyScan
from ptypy.experiment import register, utils as futils
from ptypy.utils.verbose import log
from ptypy.utils.array_utils import _translate_to_pix
f

@register()
class NXCxiPtycho(PtyScan):
    """
    First attempt to make a generalised hdf5 loader for data. Please raise a ticket in github if changes are required
    so we can coordinate. There will be a Nexus and CXI subclass to this in the future.

    Defaults:

    [name]
    default = 'Hdf5Loader'
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
            is taking place

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

    [shape]
    type = int, tuple
    default = None
    help = Shape of the region of interest cropped from the raw data.
    doc = Cropping dimension of the diffraction frame
      Can be None, (dimx, dimy), or dim. In the latter case shape will be (dim, dim).
    userlevel = 1

    """

    def __init__(self, pars=None, **kwargs):
        """
        hdf5 data loader
        """
        self.p = self.DEFAULT.copy(99)
        self.p.update(pars, in_place_depth=99)

        super(NXCxiPtycho, self).__init__(self.p, **kwargs)

        self._scantype = None
        self._ismapped = None

        self.intensities = None
        self.slow_axis = None
        self.fast_axis = None
        self.darkfield = None
        self.flatfield = None
        self.mask = None
        self.normalisation = None
        self.normalisation_laid_out_like_positions = None
        self.darkfield_laid_out_like_data = None
        self.flatfield_field_laid_out_like_data = None
        self.mask_laid_out_like_data = None

        # lets raise some exceptions here for the essentials

        log(4, u.verbose.report(self.info))
        if True in [self.p.intensities.is_swmr]:
            raise NotImplementedError("Currently swmr functionality is not implemented! Coming soon...")

        # now to populate he nexus object
        f = h5.File(self.p.intensities.file, 'r')
        nxcxi_ptycho = futils.NexusCXI(f)

        self.intensities = nxcxi_ptycho.intensity_dataset
        data_shape = self.intensities.shape
        slow_axis = nxcxi_ptycho.slow_axis_dataset[...]
        self.slow_axis = np.squeeze(slow_axis) if slow_axis.ndim>2 else slow_axis
        positions_slow_shape = self.slow_axis.shape
        fast_axis = nxcxi_ptycho.fast_axis_dataset[...]
        self.fast_axis = np.squeeze(fast_axis) if fast_axis.ndim>2 else fast_axis
        positions_fast_shape = self.fast_axis.shape
        log(3, "The shape of the \n\tdiffraction intensities is: {}\n\tslow axis data:{}\n\tfast axis data:{}".format(data_shape,
                                                                                                                       positions_slow_shape,
                                                                                                                      positions_fast_shape))
        self.compute_scan_mapping_and_trajectory(data_shape, positions_fast_shape, positions_slow_shape)

        if nxcxi_ptycho.darkfield is not None:
            self.darkfield = nxcxi_ptycho.darkfield
            log(3, "The darkfield has shape: {}".format(self.darkfield.shape))
            if self.darkfield.shape == data_shape:
                log(3, "The darkfield is laid out like the data.")
                self.darkfield_laid_out_like_data = True
            elif self.darkfield.shape == data_shape[-2:]:
                log(3, "The darkfield is not laid out like the data.")
                self.darkfield_laid_out_like_data = False
            else:
                raise RuntimeError("I have no idea what to do with this shape of darkfield data.")
        else:
            log(3, "No darkfield will be applied.")

        if nxcxi_ptycho.flatfield is not None:
            self.flatfield = nxcxi_ptycho.flatfield
            log(3, "The flatfield has shape: {}".format(self.flatfield.shape))
            if self.flatfield.shape == data_shape:
                log(3, "The flatfield is laid out like the data.")
                self.flatfield_laid_out_like_data = True
            elif self.flatfield.shape == data_shape[-2:]:
                log(3, "The flatfield is not laid out like the data.")
                self.flatfield_laid_out_like_data = False
            else:
                raise RuntimeError("I have no idea what to do with this shape of flatfield data.")
        else:
            log(3, "No flatfield will be applied.")

        if nxcxi_ptycho.pixel_mask is not None:
            self.mask = nxcxi_ptycho.pixel_mask
            log(3, "The mask has shape: {}".format(self.mask.shape))
            if self.mask.shape == data_shape:
                log(3, "The mask is laid out like the data.")
                self.mask_laid_out_like_data = True
            elif self.mask.shape == data_shape[-2:]:
                log(3, "The mask is not laid out like the data.")
                self.mask_laid_out_like_data = False
            else:
                raise RuntimeError("I have no idea what to do with this shape of mask data.")
        else:
            log(3, "No mask will be applied.")


        if nxcxi_ptycho.monitor_dataset is not None :
            self.normalisation = nxcxi_ptycho.monitor_dataset
            if (self.normalisation.shape == self.fast_axis.shape == self.slow_axis.shape):
                log(3, "The normalisation is the same dimensionality as the axis information.")
                self.normalisation_laid_out_like_positions = True
            elif self.normalisation.shape[:2] == self.fast_axis.shape == self.slow_axis.shape:
                log(3, "The normalisation matches the axis information, but will average the other dimensions.")
                self.normalisation_laid_out_like_positions = False
            else:
                raise RuntimeError("I have no idea what to do with this is shape of normalisation data.")
        else:
            log(3, "No normalisation will be applied.")

        self.p.energy = nxcxi_ptycho.energy
        self.p.distance = nxcxi_ptycho.distance
        self.p.psize = nxcxi_ptycho.x_pixel_size
        self.orientation = nxcxi_ptycho.ptypy_orientation

        # now lets figure out the cropping and centering roughly so we don't load the full data in.
        frame_shape = np.array(data_shape[-2:])

        if self.p.center is None:
            if None not in [nxcxi_ptycho.beam_center_x, nxcxi_ptycho.beam_center_y]:
                center = (nxcxi_ptycho.beam_center_x, nxcxi_ptycho.beam_center_y)
            else:
                center = frame_shape // 2 if self.p.center is None else u.expect2(self.p.center)
        center = np.array([_translate_to_pix(frame_shape[ix], center[ix]) for ix in range(len(frame_shape))])

        if self.p.shape is None:
            self.frame_slices = (slice(None, None, 1), slice(None, None, 1))
            self.p.shape = frame_shape
        else:
            pshape = u.expect2(self.p.shape)
            low_pix = center - pshape // 2
            high_pix = low_pix + pshape
            self.frame_slices = (slice(low_pix[0], high_pix[0], 1), slice(low_pix[1], high_pix[1], 1))
            self.p.center = pshape // 2 #the  new center going forward
            self.info.center = self.p.center
            self.p.shape = pshape

        # it's much better to have this logic here than in load!
        if (self._ismapped and (self._scantype is 'arb')):
            # easy peasy
            log(3, "This scan looks to be a mapped arbitrary trajectory scan.")
            self.load = self.load_mapped_and_arbitrary_scan

        if (self._ismapped and (self._scantype is 'raster')):
            log(3, "This scan looks to be a mapped raster scan.")
            self.load = self.loaded_mapped_and_raster_scan

        if (self._scantype is 'raster') and not self._ismapped:
            log(3, "This scan looks to be an unmapped raster scan.")
            self.load = self.load_unmapped_raster_scan
