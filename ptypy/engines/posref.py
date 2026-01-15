# -*- coding: utf-8 -*-
"""
Position refinement module.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
from ..core import View
from .. import utils as u
from ..utils.verbose import log
import numpy as np


class PositionRefine(object):
    def __init__(self, p):
        self.p = p

    def update_view_position(self, di_view):
        '''
        Refines the position of a single diffraction view
        Parameters
        ----------
        di_view : ptypy.core.classes.View
            A diffraction view that we wish to refine.

        Returns
        -------
        numpy.ndarray
            A length 2 numpy array with the position increments for x and y co-ordinates respectively
        '''

        raise NotImplementedError('This method needs to be overridden in order to position correct')

    def update_constraints(self, iteration):
        '''

        Parameters
        ----------
        iteration : int
            The current iteration of the engine.
        '''
        start, end = self.p.start, self.p.stop
        # Compute the maximum shift allowed at this iteration
        self.max_shift_dist = self.p.amplitude 
        if self.p.amplitude_decay:
            self.max_shift_dist *= (end - iteration) / (end - start)

    def estimate_fourier_metric(self, di_view, obj):
        '''
        Calculates error based on DM fourier error estimate.
        
        Parameters
        ----------
        di_view : ptypy.core.classes.View
            A diffraction view for which we wish to calculate the error.

        obj : numpy.ndarray
            The current calculated object for which we wish to evaluate the error against.
        Returns
        -------
        np.float
            The calculated fourier error
        '''
        af2 = np.zeros_like(di_view.data)
        for name, pod in di_view.pods.items():
            af2 += pod.downsample(u.abs2(pod.fw(pod.probe*obj)))
        return np.sum(di_view.pod.mask * (np.sqrt(af2) - np.sqrt(np.abs(di_view.data)))**2) / di_view.pod.mask.sum()

    def estimate_photon_metric(self, di_view, obj):
        '''
        Calculates error based on reduced likelihood estimate.
        
        Parameters
        ----------
        di_view : ptypy.core.classes.View
            A diffraction view for which we wish to calculate the error.

        obj : numpy.ndarray
            The current calculated object for which we wish to evaluate the error against.
        Returns
        -------
        np.float
            The calculated fourier error
        '''
        af2 = np.zeros_like(di_view.data)
        for name, pod in di_view.pods.items():
            af2 += pod.downsample(u.abs2(pod.fw(pod.probe*obj)))
        return (np.sum(di_view.pod.mask * (af2 - di_view.data)**2 / (di_view.data + 1.)) / np.prod(af2.shape))

    def cleanup(self):
        '''
        Cleans up every iteration
        '''

    @property
    def citation_dictionary(self):
        return {}


class AnnealingRefine(PositionRefine):

    def __init__(self, position_refinement_parameters, Cobj, metric="fourier"):
        '''
        Annealing Position Refinement.
        Refines the positions by the following algorithm:

        A.M. Maiden, M.J. Humphry, M.C. Sarahan, B. Kraus, J.M. Rodenburg,
        An annealing algorithm to correct positioning errors in ptychography,
        Ultramicroscopy, Volume 120, 2012, Pages 64-72
        Parameters
        ----------
        position_refinement_parameters : ptypy.utils.parameters.Param
            The parameter tree for the refinement

        Cobj : ptypy.core.classes.Container
            The current pbject container object
        metric : str
            "fourier" or "photon"
        '''
        super(AnnealingRefine, self).__init__(position_refinement_parameters)

        self.Cobj = Cobj  # take a reference here. It would be cool if we could make this read-only or something

        # Updated before each iteration by self.update_constraints
        self.max_shift_dist = None

        # Choose metric for fourier error
        if metric == "fourier":
            self.fourier_error = self.estimate_fourier_metric
        elif metric == "photon":
            self.fourier_error = self.estimate_photon_metric
        else:
            raise NotImplementedError("Metric %s is currently not implemented" %metric)

    def update_view_position(self, di_view):
        '''
        Refines the positions by the following algorithm:

        A.M. Maiden, M.J. Humphry, M.C. Sarahan, B. Kraus, J.M. Rodenburg,
        An annealing algorithm to correct positioning errors in ptychography,
        Ultramicroscopy, Volume 120, 2012, Pages 64-72

        Algorithm Description:
        Calculates random shifts around the original position and calculates the fourier error. If the fourier error
        decreased the randomly calculated postion will be used as new position.

        Parameters
        ----------
        di_view : ptypy.core.classes.View
            A diffraction view that we wish to refine.

        Returns
        -------
        numpy.ndarray
            A length 2 numpy array with the position increments for x and y co-ordinates respectively
        '''        
        # there might be more than one object view
        ob_view = di_view.pod.ob_view

        initial_coord = ob_view.coord.copy()
        coord = initial_coord
        psize = ob_view.psize.copy()

        # if you cannot move far, do nothing
        if np.max(psize) >= self.max_shift_dist:
            return np.zeros((2,))
            
        # This can be optimized by saving existing iteration fourier error...
        error = self.fourier_error(di_view, ob_view.data)
        
        for i in range(self.p.nshifts):
            # Generate coordinate shift in one of the 4 cartesian quadrants
            a, b = np.random.uniform(np.max(psize), self.max_shift_dist, 2)
            delta = np.array([(-1)**i * a, (-1)**(i//2) *b])

            if np.linalg.norm(delta) > self.p.max_shift:
                # Positions drifted too far, skip this position
                continue

            # Move view to new position
            new_coord = initial_coord + delta 
            ob_view.coord = new_coord
            ob_view.storage.update_views(ob_view)
            data = ob_view.data
            
            # catch bad slicing
            if not np.allclose(data.shape, ob_view.shape):
                continue 
                
            new_error = self.fourier_error(di_view, data)

            if new_error < error:
                # keep
                error = new_error
                coord = new_coord
                log(4, "Position correction: %s, coord: %s, delta: %s" % (di_view.ID, coord, delta))
          
        ob_view.coord = coord
        ob_view.storage.update_views(ob_view)        
        return coord - initial_coord

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

class GridSearchRefine(PositionRefine):

    def __init__(self, position_refinement_parameters, Cobj, metric="fourier"):
        '''
        Grid Search Position Refinement.

        ----------
        position_refinement_parameters : ptypy.utils.parameters.Param
            The parameter tree for the refinement

        Cobj : ptypy.core.classes.Container
            The current pbject container object
        metric : str
            "fourier" or "photon"
        '''
        super(GridSearchRefine, self).__init__(position_refinement_parameters)

        self.Cobj = Cobj  # take a reference here. It would be cool if we could make this read-only or something

        # Updated before each iteration by self.update_constraints
        self.max_shift_dist = None

        # Choose metric for fourier error
        if metric == "fourier":
            self.fourier_error = self.estimate_fourier_metric
        elif metric == "photon":
            self.fourier_error = self.estimate_photon_metric
        else:
            raise NotImplementedError("Metric %s is currently not implemented" %metric)

    def update_view_position(self, di_view):
        '''
        Refines the positions by the following algorithm:

        Calculates all shifts in a given radius around the original position and calculates the fourier error. 
        If the fourier error decreased the calculated postion will be used as new position.

        Parameters
        ----------
        di_view : ptypy.core.classes.View
            A diffraction view that we wish to refine.

        Returns
        -------
        numpy.ndarray
            A length 2 numpy array with the position increments for x and y co-ordinates respectively
        '''        
        # there might be more than one object view
        ob_view = di_view.pod.ob_view

        initial_coord = ob_view.coord.copy()
        coord = initial_coord
        psize = ob_view.psize.copy()

        # if you cannot move far, do nothing
        if np.max(psize) >= self.max_shift_dist:
            return np.zeros((2,))
            
        # This can be optimized by saving existing iteration fourier error...
        error = self.fourier_error(di_view, ob_view.data)
        
        max_shift_pix = np.ceil(self.max_shift_dist / np.min(psize))
        max_bound_pix = np.ceil(self.p.max_shift / np.min(psize))

        # Create the search grid
        deltas = np.mgrid[-max_shift_pix:max_shift_pix+1:1,
                          -max_shift_pix:max_shift_pix+1:1]
        within_bound = (deltas[0]**2 + deltas[1]**2) < (max_bound_pix**2)
        deltas = (deltas[:,within_bound] * np.min(psize)).T

        for i in range(deltas.shape[0]):
            # Current shift
            delta = deltas[i]

            # Move view to new position
            new_coord = initial_coord + delta 
            ob_view.coord = new_coord
            ob_view.storage.update_views(ob_view)
            data = ob_view.data
            
            # catch bad slicing
            if not np.allclose(data.shape, ob_view.shape):
                continue 
                
            new_error = self.fourier_error(di_view, data)

            if new_error < error:
                # keep
                error = new_error
                coord = new_coord
                log(4, "Position correction: %s, coord: %s, delta: %s" % (di_view.ID, coord, delta))
     
        ob_view.coord = coord
        ob_view.storage.update_views(ob_view)        
        return coord - initial_coord

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
