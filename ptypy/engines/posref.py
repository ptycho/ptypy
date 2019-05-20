# -*- coding: utf-8 -*-
"""
Position refinement module.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
from ..core import View
from .. import utils as u
from ..utils.verbose import log
import numpy as np
from ..core.classes import DEFAULT_ACCESSRULE


class PositionRefine(object):
    def __init__(self, p):
        self.p = p

    def update_view_position(self, di_view):
        '''
        takes a single diffraction view and updates it's co-ordinate
        '''

        raise NotImplementedError('This method needs to be overridden in order to position correct')

    def update_constraints(self, iteration):
        '''
        :param iteration:
        :return:
        updates this object based on the convergence criteria
        '''

        raise NotImplementedError('This method needs to be overridden in order to position correct')


class AnnealingRefine(PositionRefine):
    """
    Refines the positions by the following algorithm:
    
    A.M. Maiden, M.J. Humphry, M.C. Sarahan, B. Kraus, J.M. Rodenburg,
    An annealing algorithm to correct positioning errors in ptychography,
    Ultramicroscopy, Volume 120, 2012, Pages 64-72
    """
    def __init__(self, position_refinement_parameters, Cobj):
        """
        Annealing Positon Refinement.

        :param position_refinement_parameters: See parameters defined in PositionCorrectionEngine
        :param Cobj: Object Container
        """
        super(AnnealingRefine, self).__init__(position_refinement_parameters)

        self.Cobj = Cobj  # take a reference here. It would be cool if we could make this read-only or something

        # Store initial positions to check if things have drifted too far
        self.initial_positions = {}
        for oname, ob_view in Cobj.views.iteritems():
            self.initial_positions[oname] = ob_view.coord

        # Shape and pixel size
        self.shape = ob_view.shape
        self.psize = ob_view.psize[0]

        # Initialise access rules for View instantiations
        self.ar = DEFAULT_ACCESSRULE.copy()
        self.ar.psize = self.psize
        self.ar.shape = self.shape

        # Updated before each iteration by self.update_constraints
        self.max_shift_dist = None
        self.temp_ob = None
        log(3, "Position refinement initialized")

    def fourier_error(self, di_view, obj):
        """
        Calculate the Fourier error based on a given diffraction and object.

        :param di_view: View to the diffraction pattern of the given position.
        :param obj: Numpy array which contains the needed object.
        :return: Fourier Error
        """
        af2 = np.zeros_like(di_view.data)
        for name, pod in di_view.pods.iteritems():
            af2 += u.abs2(pod.fw(pod.probe*obj))
        return np.sum(di_view.pod.mask * (np.sqrt(af2) - np.sqrt(np.abs(di_view.data)))**2)

    def update_view_position(self, di_view):
        """
        Refines the positions by the following algorithm:

        A.M. Maiden, M.J. Humphry, M.C. Sarahan, B. Kraus, J.M. Rodenburg,
        An annealing algorithm to correct positioning errors in ptychography,
        Ultramicroscopy, Volume 120, 2012, Pages 64-72
    
        Algorithm Description:
        Calculates random shifts around the original position and calculates the fourier error. If the fourier error
        decreased the randomly calculated postion will be used as new position.
    
        :param di_view: Diffraction view for which a better position is searched for.

        :return The best shift found (for eventual statistics)
        """
        dcoords = np.zeros((self.p.nshifts + 1, 2)) - 1.    # Coordinates (including the original)
        delta = np.zeros((self.p.nshifts, 2))               # Coordinate shifts
        errors = np.zeros(self.p.nshifts)                   # Calculated errors for the shifted positions

        coord = di_view.pod.ob_view.coord.copy()
        
        self.ar.coord = coord
        self.ar.storageID = di_view.pod.ob_view.storageID

        # Create temporary object view that can be shifted without reformatting
        ob_view_temp = View(self.temp_ob, accessrule=self.ar)
        dcoords[0, :] = ob_view_temp.dcoord

        # This can be optimized by saving existing iteration fourier error...
        error_initial = self.fourier_error(di_view, ob_view_temp.data)

        for i in range(self.p.nshifts):
            # Generate coordinate shift in one of the 4 cartesian quadrants
            delta[i, 0] = (-1)**i * np.random.uniform(0, self.max_shift_dist)
            delta[i, 1] = (-1)**(i//2) * np.random.uniform(0, self.max_shift_dist)

            rand_coord = coord + delta[i]

            norm = np.linalg.norm(rand_coord - self.initial_positions[di_view.ID])
            if norm > self.p.max_shift:
                # Positions drifted too far, skip this position
                errors[i] = error_initial + 1.
                continue

            # Create new view on new position
            self.ar.coord = rand_coord
            ob_view_temp = View(self.temp_ob, accessrule=self.ar)
            dcoord = ob_view_temp.dcoord  # coordinate in pixel

            # Skip if this pixel has already been explored
            if any((dcoord == x).all() for x in dcoords):
                errors[i] = error_initial + 1.
                continue

            # Store new coordinate and Fourier error
            dcoords[i + 1, :] = dcoord
            errors[i] = self.fourier_error(di_view, ob_view_temp.data)

        # Identify lowest error and update position
        if np.min(errors) < error_initial:
            arg = np.argmin(errors)
            di_view.pod.ob_view.coord = coord + delta[arg]
            return delta[arg]
        else:
            return np.zeros((2,))

    def update_constraints(self, iteration):
        """
        Update information required prior to position refinement.
        :param iteration: current iteration number.
        """

        start, end = self.p.start, self.p.stop

        # Compute the maximum shift allowed at this iteration
        self.max_shift_dist = self.p.amplitude * (end - iteration) / (end - start) + self.psize/2.

        # Create a copy of the object container and expand it to avoid any run off.
        self.temp_ob = self.Cobj.copy()
        for sname, storage in self.temp_ob.storages.iteritems():
            log(4, "Old storage shape is: %s" % str(storage.shape))
            storage.padding = int(np.round(self.max_shift_dist / self.psize)) + 1
            storage.reformat()
            log(4, "New storage shape is: %s" % str(storage.shape))

    @property
    def citation_dictionary(self):
        return {
            "title" : 'An annealing algorithm to correct positioning errors in ptychography',
            "author" : 'Maiden et al.',
            "journal" : 'Ultramicroscopy',
            "volume" : 120,
            "year" : 2012,
            "page" : 64,
            "doi" : '10.1016/j.ultramic.2012.06.001',
            "comment" : 'Position Refinement using annealing algorithm'}