# -*- coding: utf-8 -*-
"""
A simple implementation of Multislice for the
ePIE algorithm.

authors: Benedikt J. Daurer and more...
"""
from ptypy.engines import stochastic
from ptypy.engines import register
from ptypy.core import geometry
from ptypy.utils import Param
from ptypy.utils.verbose import logger
from ptypy import io
import numpy as np

@register()
class ThreePIE(stochastic.EPIE):
    """
    An extension of EPIE to include multislice

    Defaults:

    [name]
    default = ThreePIE
    type = str
    help =
    doc =

    [number_of_slices]
    default = 2
    type = int
    help = The number of slices
    doc = Defines how many slices are used for the multi-slice object.

    [slice_thickness]
    default = 1e-6
    type = float, list, tuple
    help = Thickness of a single slice in meters
    doc = A single float value or a list of float values. If a single value is used, all the slice will be assumed to be of the same thickness.

    [slice_start_iteration]
    default = 0
    type = int, list, tuple
    help = iteration number to start using a specific slice
    doc =

    [fslices]
    default = slices.h5
    type = str
    help = File path for the slice data
    doc =

    """
    def __init__(self, ptycho_parent, pars=None):
        super(ThreePIE, self).__init__(ptycho_parent, pars)
        self.article = dict(
            title='{Ptychographic transmission microscopy in three dimensions using a multi-slice approach',
            author='A. M. Maiden et al.',
            journal='J. Opt. Soc. Am. A',
            volume=29,
            year=2012,
            page=1606,
            doi='10.1364/JOSAA.29.001606',
            comment='The 3PIE reconstruction algorithm',
        )
        self.ptycho.citations.add_article(**self.article)

    def engine_initialize(self):
        super().engine_initialize()

        # Create a list of objects and exit waves (one for each slice)
        self._object = [None] * self.p.number_of_slices
        self._probe = [None] * self.p.number_of_slices
        self._exits = [None] * self.p.number_of_slices
        for i in range(self.p.number_of_slices):
            self._object[i] = self.ob.copy(self.ob.ID + "_o_" + str(i))
            self._probe[i] = self.pr.copy(self.pr.ID + "_p_" + str(i))
            self._exits[i] = self.pr.copy(self.pr.ID + "_e_" + str(i))

        # ToDo:
        #    - allow for non equal slice spacing
        #    - allow for start_slice_update at a freely chosen iteration
        #      for each slice separately - works, but not if the
        #      most downstream slice is switched off

        if isinstance(self.p.slice_start_iteration, int):
            self.p.slice_start_iteration = np.ones(self.p.number_of_slices) * self.p.slice_start_iteration
        #if ĺen(self.p.slice_start_iteration) != self.p.number_of_slices:
        #    logger.info(f'dimension of given slice_start_iteration ({ĺen(self.p.slice_start_iteration)}) does not match number of slices ({self.p.number_of_slices})')

        scan = list(self.ptycho.model.scans.values())[0]
        geom = scan.geometries[0]
        g = Param()
        g.energy = geom.energy
        g.distance = self.p.slice_thickness
        g.psize = geom.resolution
        g.shape = geom.shape
        g.propagation = "nearfield"

        self.fw = []
        self.bw = []
        if type(self.p.slice_thickness) in [list, tuple]:
            assert(len(self.p.slice_thickness) == self.p.number_of_slices-1)
            for thickness in self.p.slice_thickness:
                g.distance = thickness
                G = geometry.Geo(owner=None, pars=g)
                self.fw.append(G.propagator.fw)
                self.bw.append(G.propagator.bw)
        else:
            g.distance = self.p.slice_thickness
            G = geometry.Geo(owner=None, pars=g)
            self.fw = [G.propagator.fw for i in range(self.p.number_of_slices-1)]
            self.bw = [G.propagator.bw for i in range(self.p.number_of_slices-1)]

    def engine_iterate(self, num=1):
        """
        Compute one iteration.
        """
        vieworder = list(self.di.views.keys())
        vieworder.sort()
        rng = np.random.default_rng()

        for it in range(num):

            error_dct = {}
            rng.shuffle(vieworder)

            for name in vieworder:
                view = self.di.views[name]
                if not view.active:
                    continue

                # Multislice update
                error_dct[name] = self.multislice_update(view)

            self.curiter += 1

        return error_dct

    def engine_finalize(self):
        self.ob.fill(self._object[0])
        for i in range(1, self.p.number_of_slices):
            self.ob *= self._object[i]

        # Save the slices
        slices_info = Param()
        slices_info.number_of_slices = self.p.number_of_slices
        slices_info.slice_thickness = self.p.slice_thickness
        slices_info.objects = {ob.ID: {ID: S._to_dict() for ID, S in ob.storages.items()}
                               for ob in self._object}
        slices_info.slice_start_iteration = self.p.slice_start_iteration

        header = {'description': 'multi-slices result details.'}

        h5opt = io.h5options['UNSUPPORTED']
        io.h5options['UNSUPPORTED'] = 'ignore'
        logger.info(f'Saving to {self.p.fslices}')
        io.h5write(self.p.fslices, header=header, content=slices_info)
        io.h5options['UNSUPPORTED'] = h5opt

        return super().engine_finalize()

    def multislice_update(self, view):
        """
        Performs one 'iteration' of 3PIE (multislice ePIE) for a single view.
        Based on https://doi.org/10.1364/JOSAA.29.001606
        """

        for i in range(self.p.number_of_slices-1):
            for name, pod in view.pods.items():
                # exit wave for this slice
                if self.curiter >= self.p.slice_start_iteration[i]:
                    self._exits[i][pod.pr_view] = self._probe[i][pod.pr_view] * self._object[i][pod.ob_view]
                else:
                    self._exits[i][pod.pr_view] = self._probe[i][pod.pr_view] * 1.
                # incident wave for next slice
                self._probe[i+1][pod.pr_view] = self.fw[i](self._exits[i][pod.pr_view])

        for name, pod in view.pods.items():
            # Exit wave for last slice
            if self.curiter >= self.p.slice_start_iteration[-1]:
                self._exits[-1][pod.pr_view] = self._probe[-1][pod.pr_view] * self._object[-1][pod.ob_view]
            else:
                self._exits[-1][pod.pr_view] = self._probe[-1][pod.pr_view] * 1.
            # Save final state into pod (need for ptypy fourier update)
            pod.probe = self._probe[-1][pod.pr_view]
            pod.object = self._object[-1][pod.ob_view]
            pod.exit = self._exits[-1][pod.pr_view]

        # Fourier update
        error = self.fourier_update(view)

        # Object/probe update for the last slice
        if self.curiter >= self.p.slice_start_iteration[-1]:
            self.object_update(view, {pod.ID:self._exits[-1][pod.pr_view] for name, pod in view.pods.items()})
            self.probe_update(view, {pod.ID:self._exits[-1][pod.pr_view] for name, pod in view.pods.items()})
            for name, pod in view.pods.items():
                self._object[-1][pod.ob_view] = pod.object
                self._probe[-1][pod.pr_view] = pod.probe
        else:
            for name, pod in view.pods.items():
                self._probe[-1][pod.pr_view] = pod.exit * 1.

        # Object/probe update for other slices (backwards)
        for i in range(self.p.number_of_slices-2, -1, -1):
            if self.curiter >= self.p.slice_start_iteration[i]:

                for name, pod in view.pods.items():
                    # Backwards propagation of the probe
                    pod.exit = self.bw[i](self._probe[i+1][pod.pr_view])
                    # Save state into pods
                    pod.probe = self._probe[i][pod.pr_view]
                    pod.object = self._object[i][pod.ob_view]

                # Actual object/probe update
                self.object_update(view, {pod.ID:self._exits[i][pod.pr_view] for name, pod in view.pods.items()})
                self.probe_update(view, {pod.ID:self._exits[i][pod.pr_view] for name, pod in view.pods.items()})
                for name, pod in view.pods.items():
                    self._object[i][pod.ob_view] = pod.object
                    self._probe[i][pod.pr_view] = pod.probe
            else:
                for name, pod in view.pods.items():
                    self._probe[i][pod.pr_view] = self.bw[i](self._probe[i+1][pod.pr_view])

        # set the object as the product of all slices for better live plotting
        self.ob.fill(self._object[0])
        for i in range(1, self.p.number_of_slices):
            self.ob *= self._object[i]

        return error