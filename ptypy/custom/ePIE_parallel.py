# -*- coding: utf-8 -*-
"""
Parallel ePIE reconstruction engine.

This is an implementation of "Synchronous parallel ptychography" as
described in Nashed et al. [Optics Express, 22 (2014) 32082]. Note
that this algorithm is not a strict parallelization of the original
ePIE algorithm. The number of nodes affects reconstruction as
described in the publication.

This class does not carry out the slimmed object sharing described in
Nashed et al., but instead shares the entire object array as done in
for example the PTYPY implementation of the Differece Map algorithm.

Note that these PTYPY-specific reconstruction options are not
(yet) implemented:
* subpixel stuff
* log likelihood / photon errors

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
import time
import random

from ptypy import utils as u
from ptypy.utils.verbose import logger
from ptypy.utils import parallel
from ptypy.engines import BaseEngine, register
from ptypy.core.manager import Full, Vanilla

__all__ = ['EPIEParallel']


@register(name = 'ePIEparallel')
class EPIEParallel(BaseEngine):
    """
    Parallel ePIE reconstruction engine.


    Defaults:

    [name]
    default = ePIEparallel
    type = str
    help =
    doc =

    [alpha]
    default = 1.
    type = float
    lowlim = 0.0
    uplim = 1.0
    help = ePIE object update parameter
    doc = Step size for the object update, a higher value will give faster change.

    [beta]
    default = 1.
    type = float
    lowlim = 0.0
    uplim = 1.0
    help = ePIE probe update parameter
    doc = Step size for the probe update, a higher value will give faster change.

    [probe_update_start]
    default = 2
    type = int
    lowlim = 0
    help = Number of iterations before probe update starts

    [synchronization]
    default = 1
    type = int
    lowlim = 1
    help = Probe/object synchronization period
    doc = Period with which to synchronize the object (and optionally the probe) among parallel nodes.

    [redistribute_data]
    default = True
    type = bool
    help = Redistribute views to form blocks
    doc = Whether or not to redistribute data among nodes to keep each node's views in a contiguous geographic block, even if new data is added during reconstruction.

    [average_probe]
    default = False
    type = bool
    help = Average probe among nodes
    doc = Whether or not to average the probe among nodes, otherwise each node has its own probe as in the original publication. Averaging seems to work the best.

    [random_order]
    default = True
    type = bool
    help = Visit positions in random order
    doc = Whether to cycle through the positions in random order on each ePIE iteration. Otherwise does the pods in alphabetical order as per list.sort(). Disabling is useful for debugging.

    [clip_object]
    default = None
    type = tuple
    help = Clip object amplitude into this interval

    [obj_smooth_std]
    default = None
    type = int
    lowlim = 0
    help = Gaussian smoothing (pixel) of the current object prior to update
    doc = If None, smoothing is deactivated. This smoothing can be used to reduce the amplitude of spurious pixels in the outer, least constrained areas of the object.

    [probe_center_tol]
    default = 3
    type = float
    lowlim = 0.0
    help = Pixel radius around optical axes that the probe mass center must reside in

    [compute_log_likelihood]
    default = True
    type = bool
    help = A switch for computing the log-likelihood error

    """

    SUPPORTED_MODELS = [Full, Vanilla]

    def __init__(self, ptycho_parent, pars=None):
        """
        ePIE reconstruction engine.
        """
        super().__init__(ptycho_parent, pars)

        p = self.DEFAULT.copy()
        if pars is not None:
            p.update(pars)
        self.p = p

        # Check that smoothing doesn't outrun object sharing
        if ((self.p.obj_smooth_std is not None) and
                (self.p.synchronization > 1)):
            logger.warning(
                'Object smoothing with intermittent synchronization (synchronization > 1) usually causes total blurring.')

        # Instance attributes
        self.ob_nodecover = None
        self.mean_power = None

        self.ptycho.citations.add_article(
            title='An improved ptychographical phase retrieval algorithm for diffractive imaging',
            author='Maiden A. and Rodenburg J.',
            journal='Ultramicroscopy',
            volume=10,
            year=2009,
            page=1256,
            doi='10.1016/j.ultramic.2009.05.012',
            comment='The ePIE reconstruction algorithm',
        )

    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        # we need the "node coverage" - the number of processes which
        # has views onto each pixel of the object. Could be a simple
        # int array but guess it's best to use containers to allow for
        # more than one object storage (as in DM).
        self.ob_nodecover = self.ob.copy(self.ob.ID + '_ncover', fill=0.0)

    def engine_prepare(self):
        """
        Last minute initialization. Everything that needs to be
        recalculated when new data arrives.
        """
        if self.p.redistribute_data:
            self._redestribute_data()

        # mark the pixels covered per node
        self.ob_nodecover.fill(0.0)
        for name, pod in self.pods.items():
            if pod.active:
                self.ob_nodecover[pod.ob_view] = 1
        self.nodemask = np.array(list(self.ob_nodecover.S.values())[0].data[0],
                                 dtype=np.bool)

        # communicate this over MPI
        parallel.allreduceC(self.ob_nodecover)

        # Mean power in the data
        mean_power = 0.
        for name, s in self.di.storages.items():
            mean_power += s.mean_power
        self.mean_power = mean_power / len(self.di.storages)


        # DEBUGGING: show the actual domain decomposition
        # if self.curiter == 0:
        #     import matplotlib.pyplot as plt
        #     plt.imshow(self.nodemask)
        #     plt.colorbar()
        #     plt.show()
        #     if parallel.master:
        #         import matplotlib.pyplot as plt
        #         plt.imshow(list(self.ob_nodecover.S.values())[0].data[0].real)
        #         plt.colorbar()
        #         plt.show()

    def engine_iterate(self, num=1):
        """
        Compute `num` iterations.
        """
        pod_order = list(self.pods.keys())
        pod_order.sort()
        to = 0.0
        tf = 0.0
        tc = 0.0
        for it in range(num):
            pre_str = 'Iteration %u:  ' % it
            error_dct = {}
            if self.p.random_order:
                random.shuffle(pod_order)
            do_update_probe = (self.p.probe_update_start <= self.curiter + it)

            # object smooting prior to update, if requested
            if self.p.obj_smooth_std is not None:
                for name, s in self.ob.S.items():
                    # u.c_gf is a complex wrapper around
                    # scipy.ndimage.gaussian_filter()
                    std = self.p.obj_smooth_std
                    s.data[:] = u.c_gf(s.data, [0, std, std])

            for name in pod_order:
                pod = self.pods[name]
                if not pod.active:
                    continue

                # Fourier update:
                logger.debug(pre_str + '----- ePIE fourier update -----')
                t0 = time.time()
                exit_ = pod.object * pod.probe
                image = pod.fw(exit_)
                fmag = np.sqrt(np.abs(pod.diff))
                error_fmag = (
                    np.sum(pod.mask * (pod.downsample(np.abs(image)) - fmag)**2)
                    / pod.mask.sum()
                )
                image = (
                    pod.upsample(pod.mask * fmag) * np.exp(1j * np.angle(image))
                    + pod.upsample(1 - pod.mask) * image
                )
                pod.exit = pod.bw(image)

                error_exit = np.sum(np.abs(pod.exit - exit_)**2)
                if self.p.compute_log_likelihood:
                    LL = pod.downsample(u.abs2(pod.fw(pod.probe * pod.object)))
                    error_phot = (np.sum(pod.mask * (LL - pod.diff)**2 / (pod.diff + 1.)) / np.prod(LL.shape))
                else:
                    error_phot = 0.
                error_dct[name] = [error_fmag, error_phot, error_exit]

                t1 = time.time()
                tf += t1 - t0

                # Power correection
                # scale probe such that its mean power equals the mean power of the diffraction data
                # This stabilizes the ePIE algorithm and prevents the probe from growing too large.
                pod.probe *= np.sqrt(self.mean_power / u.abs2(pod.probe).mean())

                # Object update:
                logger.debug(pre_str + '----- ePIE object update -----')
                pod.object += (self.p.alpha
                               * np.conj(pod.probe)
                               / np.max(np.abs(pod.probe) ** 2)
                               * (pod.exit - exit_))

                # Probe update: The ePIE paper (and the parallel ePIE
                # paper) are unclear as to what maximum value should be
                # chosen here. An alternative would be the maximum of
                # np.abs(pod.object)**2, but that tends to explode the
                # probe.
                if do_update_probe:
                    logger.debug(pre_str + '----- ePIE probe update -----')
                    object_max = np.max(
                        np.abs(list(self.ob.S.values())[0].data.max())**2)
                    pod.probe += (self.p.beta
                                  * np.conj(pod.object) / object_max
                                  * (pod.exit - exit_))
                    # Apply the probe support
                    if self._probe_support:
                        pod.probe *= self._probe_support[pod.pr_view.storageID][0]
                t2 = time.time()
                to += t2 - t1

            # center the probe, if requested
            self.center_probe()

            # clip the object, if requested
            if self.p.clip_object is not None:
                low, high = self.p.clip_object
                for name, s in self.ob.S.items():
                    phase = np.angle(s.data)
                    ampl = np.abs(s.data)
                    under = (ampl < low)
                    over = (ampl > high)
                    clipped = under.sum() + over.sum()
                    if clipped:
                        logger.info(
                            'Clipping the object in %u pixels' % clipped)
                    s.data[under] = low * np.exp(1j * phase[under])
                    s.data[over] = high * np.exp(1j * phase[over])

            # Distribute result with MPI
            if (self.curiter + it) % self.p.synchronization == 0:
                logger.debug(pre_str + '----- communication -----')

                # only share the part of the object which whis node has
                # contributed to, and zero the rest to avoid weird
                # feedback.
                list(self.ob.S.values())[0].data[0] *= self.nodemask
                parallel.allreduceC(self.ob)

                # the reduced sum should be an average, and the
                # denominator (the number of contributing nodes) varies
                # across the object.
                for name, s in self.ob.S.items():
                    s.data /= (np.abs(self.ob_nodecover.S[name].data) + 1e-5)

                # average the probe across nodes, if requested
                if self.p.average_probe and do_update_probe:
                    for name, s in self.pr.S.items():
                        parallel.allreduce(s.data)
                        s.data /= parallel.size
                t3 = time.time()
                tc += t3 - t2

            self.curiter += 1

        logger.info('Time spent in Fourier update: %.2f' % tf)
        logger.info('Time spent in Overlap update: %.2f' % to)
        logger.info('Time spent in communication:  %.2f' % tc)

        # error_dct is in the format that basic_fourier_update returns
        # and that Ptycho expects. In DM, that dict is overwritten on
        # every iteration, so we only gather the dicts corresponding to
        # the last iteration of each contiguous block.
        return error_dct

    def engine_finalize(self):
        """
        Try deleting every helper container.
        """
        containers = [self.ob_nodecover, ]

        for c in containers:
            logger.debug('Attempt to remove container %s' % c.ID)
            del self.ptycho.containers[c.ID]
        #    IDM.used.remove(c.ID)

        del containers

    def _redestribute_data(self):
        """
        This function redistributes data among nodes, so that each
        node becomes in charge of a contiguous block of scanning
        positions.

        Each node is associated with a domain of the scanning pattern,
        and communication happens node-to-node after each has worked
        out which of its pods are not part of its domain.

        """
        layout = self._best_decomposition(parallel.size)
        t0 = time.time()

        # get the range of positions and define the size of each node's domain
        pod = list(self.pods.values())[0]
        xlims = [pod.ob_view.coord[1], ] * 2  # min, max
        ylims = [pod.ob_view.coord[0], ] * 2  # min, max
        for name, pod in self.pods.items():
            xlims = [min(xlims[0], pod.ob_view.coord[1]),
                     max(xlims[1], pod.ob_view.coord[1])]
            ylims = [min(ylims[0], pod.ob_view.coord[0]),
                     max(ylims[1], pod.ob_view.coord[0])]
        # expand the outer borders slightly to avoid edge effects
        xlims = np.array(xlims) + np.array([-1, 1]) * np.diff(xlims) * .001
        ylims = np.array(ylims) + np.array([-1, 1]) * np.diff(ylims) * .001
        # the domains sizes
        dx = np.diff(xlims) / layout[1]
        dy = np.diff(ylims) / layout[0]

        # now, the node number corresponding to a coordinate (x, y) is
        def __node(x, y):
            return (int((x - xlims[0]) / dx)
                    + layout[1] * int((y - ylims[0]) / dy))

        # now, each node works out which of its own pods to send off,
        # and the result is communicated to all other nodes as a dict.
        destinations = {}
        for name, pod in self.pods.items():
            if not pod.active:
                continue
            y, x = pod.ob_view.coord
            if not __node(x, y) == parallel.rank:
                destinations[name] = __node(x, y)
        destinations = parallel.gather_dict(destinations)
        destinations = parallel.bcast_dict(destinations)
        if len(destinations.keys()) == 0:
            return 0

        # prepare (enlarge) the storages on the receiving nodes
        sendpods = []
        for name, dest in destinations.items():
            if self.pods[name].active:
                # sending this pod, so add it to a temporary list
                sendpods.append(name)

        for name, dest in destinations.items():
            if dest == parallel.rank:
                # receiving this pod, so mark it as active
                self.pods[name].di_view.active = True
                self.pods[name].ma_view.active = True
                self.pods[name].ex_view.active = True
        for name in ['Cdiff', 'Cmask']:
            self.ptycho.containers[name].reformat()

        # transfer data
        transferred = 0
        for name, dest in destinations.items():
            if name in sendpods:
                # your turn to send
                parallel.send(self.pods[name].diff, dest=dest)
                parallel.send(self.pods[name].mask, dest=dest)
                self.pods[name].di_view.active = False
                self.pods[name].ma_view.active = False
                self.pods[name].ex_view.active = False
                transferred += 1
            if dest == parallel.rank:
                # your turn to receive
                self.pods[name].diff = parallel.receive()
                self.pods[name].mask = parallel.receive()
            parallel.barrier()
        for name in ['Cdiff', 'Cmask', 'Cexit']:
            self.ptycho.containers[name].reformat()
        transferred = parallel.comm.reduce(transferred)
        t1 = time.time()

        if parallel.master:
            logger.info('Redistributed data to match %ux%u grid, moved %u pods in %.2f s'
                        % (tuple(layout) + (transferred, t1 - t0)))

        return (t1 - t0)

    def _best_decomposition(self, N):
        """
        Work out the best arrangement of domains for a given number of
        nodes. Assumes a roughly square scan.
        """
        solutions = []
        for i in range(1, int(np.sqrt(N)) + 1):
            if N % i == 0:
                solutions.append(i)
        i = max(solutions)
        assert (i * (N // i) == N)
        return [i, N // i]

    def center_probe(self):
        """
        Stolen in its entirety from the DM engine.
        """
        if self.p.probe_center_tol is not None:
            for name, pr_s in self.pr.storages.items():
                c1 = u.mass_center(u.abs2(pr_s.data).sum(0))
                c2 = np.asarray(pr_s.shape[-2:]) // 2
                # fft convention should however use geometry instead
                if u.norm(c1 - c2) < self.p.probe_center_tol:
                    break
                # SC: possible BUG here, wrong input parameter
                pr_s.data[:] = u.shift_zoom(pr_s.data, (1.,)*3,
                        (0, c1[0], c1[1]), (0, c2[0], c2[1]))

                # shift the object
                ob_s = pr_s.views[0].pod.ob_view.storage
                ob_s.data[:] = u.shift_zoom(ob_s.data, (1.,)*3,
                        (0, c1[0], c1[1]), (0, c2[0], c2[1]))

                # shift the exit waves, loop through different exit wave views
                for pv in pr_s.views:
                    pv.pod.exit = u.shift_zoom(pv.pod.exit, (1.,)*2,
                            (c1[0], c1[1]), (c2[0], c2[1]))

                logger.info('Probe recentered from %s to %s' %
                            (str(tuple(c1)), str(tuple(c2))))
