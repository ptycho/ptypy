"""
An implementation of the Regularised Average Successive Projections (RASP)
ptychographic algorithm

Authors: Andy Maiden
"""
import numpy as np

from ..engines import projectional, register
from ..core import geometry
from ..utils import Param
from ..utils.verbose import logger
from .. import io
from .. import utils as u


@register()
class RASP(projectional._ProjectionEngine):
    """
    Regularised Average Successive Projections

    Defaults:

    [name]
    default = RASP
    type = str
    help =
    doc =

    [alpha]
    default = 1.
    type = float
    lowlim = 0.0
    help = object step parameter

    [beta]
    default = 1.
    type = float
    lowlim = 0.0
    help = probe step parameter

    [probe_power_correction]
    default = True
    type = bool
    help = A switch to correct probe power
    """

    def __init__(self, ptycho_parent, pars=None):
        super().__init__(ptycho_parent, pars)

        self.article = dict(
                title='Regularised Average Successive Projections: A Brilliant Ptychographyic Algorithm',
                author='A. M. Maiden et al.',
                journal='Journal',
                volume=42,
                year=2023,
                page=42,
                doi='doi',
                comment='Regularised Average Successive Projections',
                )
        self.ptycho.citations.add_article(**self.article)

    def engine_initialize(self):
        super().engine_initialize()

        self.topSumO = {}
        self.botSumO = {}
        self.topSumP = {}
        self.botSumP = {}

        # for d in self.di.storages.values():
            # d.data = np.fft.fftshift(np.sqrt(d.data))

        if self.p.probe_power_correction:
            self.probe_power_correction()

        # force clipping
        self.p.clip_object = (0, 1.)

    def probe_power_correction(self):
        # find probe power from brightest diffraction pattern
        probe_power = 0
        for d in self.di.storages.values():
            max_ind = np.argmax(np.sum(d.data, axis=(1,2)))
            current_pp = np.sum(d.data[max_ind, :, :])
            if current_pp > probe_power:
                probe_power = current_pp

        # correct the initial probe's power
        for p in self.pr.storages.values():
            p.data = p.data * np.sqrt(probe_power / (p.data.size * np.sum(u.abs2(p.data))))

    def overlap_update(self):

        vieworder = list(self.di.views.keys())
        # TODO: why do I need sort here?
        # vieworder.sort()
        rng = np.random.default_rng()

        # reset the accumulated sum of object/probe before going through all
        # the diffraction view for this iteration
        for name, s in self.ob.storages.items():
            self.topSumO[name] = np.zeros_like(s.data)
            self.botSumO[name] = np.zeros_like(s.data)
        for name, p in self.pr.storages.items():
            self.topSumP[name] = np.zeros_like(p.data)
            self.botSumP[name] = np.zeros_like(p.data)

        rng.shuffle(vieworder)

        for name in vieworder:
            view = self.di.views[name]
            if not view.active:
                continue

            # RASP
            self.rasp_update(view)

        # averaging
        self.object_update()
        self.probe_update()

        # Recenter the probe
        self.center_probe()

    def object_update(self):
        for name, s in self.ob.storages.items():
            eps = np.finfo(self.botSumO[name].dtype).eps
            s.data = self.topSumO[name] / (self.botSumO[name] + eps)
            self.clip_object(s)

    def probe_update(self):
        for name, p in self.pr.storages.items():
            eps = np.finfo(self.botSumP[name].dtype).eps
            p.data = self.topSumP[name] / (self.botSumP[name] + eps)

    def clip_object(self, ob):
        # override the clipping
        if self.p.clip_object is not None:
            _, clip_max = self.p.clip_object
            ampl_obj = np.abs(ob.data)
            too_high = (ampl_obj > clip_max)
            ob.data[too_high] = clip_max * u.sign(ob.data[too_high])

    def rasp_update(self, view):

        # name of this object/probe
        ob_name = view.pod.ob_view.storageID
        pr_name = view.pod.pr_view.storageID

        # the global object slice of this view
        ob_slice = view.pod.ob_view.slice
        pr_slice = view.pod.pr_view.slice

        # local object/probe
        objBox = view.pod.object
        probeBox = view.pod.probe
        conjO = np.conj(objBox)
        conjP = np.conj(probeBox)

        # update wave modulus using measured data
        absO2 = u.abs2(objBox)
        absP2 = u.abs2(probeBox)
        EW = probeBox * objBox
        # TODO: which fft to use?
        dd = np.sqrt(np.fft.fftshift(view.pod.diff))
        revisedEW = np.fft.ifft2(dd * u.sign(np.fft.fft2(EW)))
        deltaEW = revisedEW - EW

        # update global object/probe
        view.pod.ob_view.data = objBox + 0.5 * conjP * deltaEW / (np.mean(absP2)*self.p.alpha + absP2)
        view.pod.pr_view.data = probeBox + conjO * deltaEW / (self.p.beta + absO2)

        self.topSumO[ob_name][ob_slice] += conjP * revisedEW
        self.botSumO[ob_name][ob_slice] += absP2
        self.topSumP[pr_name] += conjO * revisedEW
        self.botSumP[pr_name] += absO2
