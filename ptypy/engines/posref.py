# -*- coding: utf-8 -*-
"""
Position refinement module.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
from ..core import View
from ..core.classes import DEFAULT_ACCESSRULE
from .. import utils as u
from ..utils import parallel
from ..utils.verbose import log, logger
import numpy as np
import os
import sys
import time
import gc

class PositionRefine(object):
    """
    Refines the positions by the following algorithm:
    
    A.M. Maiden, M.J. Humphry, M.C. Sarahan, B. Kraus, J.M. Rodenburg,
    An annealing algorithm to correct positioning errors in ptychography,
    Ultramicroscopy, Volume 120, 2012, Pages 64-72
    """
    def __init__(self, engine):
        self.engine = engine
        self.number_rand_shifts = engine.p.position_refinement.nshifts # should be a multiple of 4
        self.amplitude = engine.p.position_refinement.amplitude        # still has to be replaced by parameter value
        self.max_shift_allowed = engine.p.position_refinement.max_shift

        # Keep track of the initial positions
        self.initial_pos = np.zeros((len(self.engine.di.views),2))
        di_view_order = self.engine.di.views.keys()
        di_view_order.sort()
        for i, name in enumerate(di_view_order):
            di_view = self.engine.di.views[name]
            self.initial_pos[i, 0] = di_view.pod.ob_view.coord[0]
            self.initial_pos[i, 1] = di_view.pod.ob_view.coord[1]

        # Shape and pixelsize 
        self.shape = self.engine.pr.S.values()[0].data[0].shape
        self.psize = self.engine.ob.S.values()[0].psize[0]

        # Maximum shift
        start, end = engine.p.position_refinement.start, engine.p.position_refinement.stop
        self.max_shift_dist_rule = lambda it: self.amplitude * (end - it) / (end - start) + self.psize/2.

        # Save initial positions to file
        self.save_pos()
        logger.info("Position refinement initialized")


    def fourier_error(self, di_view, obj):
        """
        Calculates the fourier error based on a given diffraction and ohject.

        :param di_view: View to the diffraction pattern of the given position.
        :param object: Numpy array which contains the needed object.
        :return: Tuple of Fourier Errors
        """
        af2 = np.zeros_like(di_view.data)
        for name, pod in di_view.pods.iteritems():
            af2 += u.abs2(pod.fw(pod.probe*obj))
        af = np.sqrt(af2)
        fmag = np.sqrt(np.abs(di_view.data))
        error = np.sum(di_view.pod.mask * (af - fmag)**2)
        del af2
        del af
        del fmag
        return error

    def update(self):
        """
        Iterates trough all positions and refines them by a given algorithm. 
        Right now the following algorithm is implemented:
        
        A.M. Maiden, M.J. Humphry, M.C. Sarahan, B. Kraus, J.M. Rodenburg,
        An annealing algorithm to correct positioning errors in ptychography,
        Ultramicroscopy, Volume 120, 2012, Pages 64-72
        """
        log(4, "----------- START POS REF -------------")        
        di_view_names = self.engine.di.views.keys()
        
        # List of refined coordinates which will be used to reformat the object
        new_coords = np.zeros((len(di_view_names), 2))

        # Only used for calculating the shifted pos
        self.engine.temp_ob = self.engine.ob.copy()

        # Maximum shift
        self.max_shift_dist = self.max_shift_dist_rule(self.engine.curiter)

        # Iterate through all diffraction views
        for i, di_view_name in enumerate(self.engine.di.views):
            di_view = self.engine.di.views[di_view_name]
            pos_num = int(di_view.ID[1:])
            
            # create accessrule
            if i == 0:
                self.engine.ar = DEFAULT_ACCESSRULE.copy()
                self.engine.ar.psize = self.engine.temp_ob.storages.values()[0].psize
                self.engine.ar.shape = self.shape
            
            # Check for new coordinates
            if di_view.active:
                new_coords[pos_num, :] = self.single_pos_ref(di_view)

        # MPI reduce and update new coordinates
        new_coords = parallel.allreduce(new_coords)
        for di_view_name in self.engine.di.views:
            di_view = self.engine.di.views[di_view_name]
            pos_num = int(di_view.ID[1:])
            if new_coords[pos_num, 0] != 0 and new_coords[pos_num, 1] != 0:
                log(4, "Old coordinate (%d): " %(pos_num) + str(di_view.pod.ob_view.coord))
                log(4, "New coordinate (%d): " %(pos_num) + str(new_coords[pos_num, :]))
                di_view.pod.ob_view.coord = new_coords[pos_num, :]

        # Update object based on new position coordinates
        self.engine.ob.reformat()
        
        # The size of the object might have been changed
        del self.engine.ptycho.containers[self.engine.ob.ID + '_vcover']
        del self.engine.ptycho.containers[self.engine.ob.ID + '_nrm']
        self.engine.ob_viewcover = self.engine.ob.copy(self.engine.ob.ID + '_vcover', fill=0.)
        self.engine.ob_nrm = self.engine.ob.copy(self.engine.ob.ID + '_nrm', fill=0.)
        for name, s in self.engine.ob_viewcover.storages.iteritems():
            s.fill(s.get_view_coverage())

        # clean up
        del self.engine.ptycho.containers[self.engine.temp_ob.ID]
        del self.engine.temp_ob
        gc.collect()

    def single_pos_ref(self, di_view):
        """
        Refines the positions by the following algorithm:

        A.M. Maiden, M.J. Humphry, M.C. Sarahan, B. Kraus, J.M. Rodenburg,
        An annealing algorithm to correct positioning errors in ptychography,
        Ultramicroscopy, Volume 120, 2012, Pages 64-72
    
        Algorithm Description:
        Calculates random shifts around the original position and calculates the fourier error. If the fourier error
        decreased the randomly calculated postion will be used as new position.
    
        :param di_view: Diffraction view for which a better position is searched for.

        :return: If a better coordinate (smaller fourier error) is found for a position, the new coordinate (meters)
        will be returned. Otherwise (0, 0) will be returned.
        """
        dcoords = np.zeros((self.number_rand_shifts + 1, 2)) - 1.
        delta = np.zeros((self.number_rand_shifts, 2))               # coordinate shift
        errors = np.zeros(self.number_rand_shifts)                   # calculated error for the shifted position
        coord = np.copy(di_view.pod.ob_view.coord)
        
        self.engine.ar.coord = coord
        self.engine.ar.storageID = self.engine.temp_ob.storages.values()[0].ID

        # Create temporal object view that can be shifted without reformatting
        ob_view_temp = View(self.engine.temp_ob, accessrule=self.engine.ar)
        dcoords[0, :] = ob_view_temp.dcoord

        # This can be optimized by saving existing iteration fourier error...
        error_inital = self.fourier_error(di_view, ob_view_temp.data)

        for i in range(self.number_rand_shifts):
            delta_y = np.random.uniform(0, self.max_shift_dist)
            delta_x = np.random.uniform(0, self.max_shift_dist)

            if i % 4 == 1:
                delta_y *= -1
                delta_x *= -1
            elif i % 4 == 2:
                delta_x *= -1
            elif i % 4 == 3:
                delta_y *= -1

            delta[i, 0] = delta_y
            delta[i, 1] = delta_x

            rand_coord = [coord[0] + delta_y, coord[1] + delta_x]
            norm = np.linalg.norm(rand_coord - self.initial_pos[int(di_view.ID[1:]), :])

            if norm > self.max_shift_allowed:
                # Positions drifted too far, skip this position
                #log(4, "New position is too far away!!!", parallel=True)
                errors[i] = error_inital + 1.
                continue

            self.engine.ar.coord = rand_coord
            ob_view_temp = View(self.engine.temp_ob, accessrule=self.engine.ar)
            dcoord = ob_view_temp.dcoord  # coordinate in pixel

            # check if new coordinate is on a different pixel since there is no subpixel shift, if there is no shift
            # skip the calculation of the fourier error
            if any((dcoord == x).all() for x in dcoords):
                errors[i] = error_inital + 1.
                continue
            dcoords[i + 1, :] = dcoord

            # Check if these are really necessary
            if di_view.pod.ob_view.dlow[0] < 0:
                di_view.pod.ob_view.dlow[0] = 0
            if di_view.pod.ob_view.dlow[1] < 0:
                di_view.pod.ob_view.dlow[1] = 0

            new_obj = np.zeros(self.shape, dtype=np.complex128)
            if ob_view_temp.data.shape[0] != self.shape[0] or ob_view_temp.data.shape[1] != self.shape[1]:
                # if the data of the view has the wrong shape, zero-pad the data
                # new data for calculating the fourier transform
                # calculate limits of the grid
                ymin = self.engine.ob.storages.values()[0].grids()[0][0, 0, 0]
                ymax = self.engine.ob.storages.values()[0].grids()[0][0, -1, -1]
                xmin = self.engine.ob.storages.values()[0].grids()[1][0, 0, 0]
                xmax = self.engine.ob.storages.values()[0].grids()[1][0, -1, -1]

                # check if the new array would be bigger
                new_xmin = rand_coord[1] - (self.psize * self.shape[1] / 2.)
                new_xmax = rand_coord[1] + self.psize * self.shape[1] / 2.
                new_ymin = rand_coord[0] - (self.psize * self.shape[0] / 2.)
                new_ymax = rand_coord[0] + self.psize * self.shape[0] / 2.
        
                idx_x_low = 0
                idx_x_high = self.shape[1]
                idx_y_low = 0
                idx_y_high = self.shape[0]

                if new_ymin < ymin:
                    idx_y_low = self.shape[0] - ob_view_temp.data.shape[0]
                elif new_ymax > ymax:
                    idx_y_high = ob_view_temp.data.shape[0]

                if new_xmin < xmin:
                    idx_x_low = self.shape[1] - ob_view_temp.data.shape[1]
                elif new_xmax > xmax:
                    idx_x_high = ob_view_temp.data.shape[1]

                new_obj[idx_y_low: idx_y_high, idx_x_low: idx_x_high] = ob_view_temp.data
            else:
                new_obj = ob_view_temp.data

            errors[i] = self.fourier_error(di_view, new_obj)
            del new_obj

        if np.min(errors) < error_inital:
            # if a better coordinate is found
            #log(4, "New coordinate with smaller Fourier Error found!", parallel=True)
            arg = np.argmin(errors)
            new_coordinate = np.array([coord[0] + delta[arg, 0], coord[1] + delta[arg, 1]])
        else:
            new_coordinate = (0, 0)

        # Clean up
        del ob_view_temp
        del di_view
        
        return new_coordinate

    def save_pos(self):
        """
        Saves the current positions in a .txt file.
        """
        if parallel.master:
            coords = []

            di_view_names = self.engine.di.views.keys()
            di_view_names.sort()
            for di_view_name in di_view_names:
                di_view = self.engine.di.views[di_view_name]
                coord = np.copy(di_view.pod.ob_view.coord)
                coords.append(coord)
            coords = np.asarray(coords)

            directory = "positions_" + sys.argv[0][:-3] + "/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            np.savetxt(directory + "pos_" + str(self.engine.p.name) + "_" + str(self.engine.curiter).zfill(4) + ".txt", coords)
