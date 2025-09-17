# -*- coding: utf-8 -*-
"""
Modified version of the DM engine to work with scanning mirror data.

authors: Ken V. Falch
"""
from ptypy.engines import projectional
from ptypy.engines import register
from ptypy import utils as u
from ptypy.core.manager import ScanningMirrorModel
from ptypy.engines.posref import AnnealingRefine, GridSearchRefine
import numpy as np


__all__ = ['DM_scanning_mirror']


def _projection_update_generalized(diff_view, a, b, c, pbound=None):
    """
    Modified version of en engines.utils.projection_update_generalized.

    Parameters
    ----------
    diff_view : View
        View to diffraction data

    a,b,c : float
        Coefficients for Overlap, Fourier and Fourier * Overlap constraints,
        respectively

    pbound : float, optional
        Power bound. Fourier update is bypassed if the quadratic deviation
        between diffraction data and `diff_view` is below this value.
        If ``None``, fourier update always happens.

    Returns
    -------
    err_fmag, err_exit : float

        - `err_fmag`, Fourier magnitude error; quadratic deviation from
          root of experimental data
        - `err_exit`, quadratic deviation between exit waves before and after
          projection
    """
    # Prepare dict for storing propagated waves
    f = {}

    # Buffer for accumulated photons
    af2 = np.zeros_like(diff_view.data)
    # Get measured data
    I = diff_view.data

    # Get the mask (cast to the same type as diff, for precision when operating
    # with other numerical arrays)
    fmask = diff_view.pod.mask.astype(I.dtype)

    # Propagate the exit waves
    for name, pod in diff_view.pods.items():
        if not pod.active:
            continue
        f[name] = pod.fw((1-c) * pod.exit + c * pod.probe * pod.object)
        af2 += pod.downsample(u.abs2(f[name]))

    fmag = np.sqrt(np.abs(I))
    af = np.sqrt(af2)

    # Fourier magnitudes deviations
    fdev = af - fmag
    err_fmag = np.sum(fmask * fdev**2) / fmask.sum()
    err_exit = 0.

    if pbound is None:
         fm = (1 - fmask) + fmask * fmag / (af + 1e-10)
    elif err_fmag > pbound:
         renorm = np.sqrt(pbound / err_fmag)
         fm = (1 - fmask) + fmask * (fmag + fdev * renorm) / (af + 1e-10)
    else:
         fm = None

    for name, pod in diff_view.pods.items():
        if not pod.active:
            continue

        if fm is not None:
            df = b * pod.bw(pod.upsample(fm) * f[name]) + \
                 a * pod.probe * pod.object - (a + b) * pod.exit
        else:
            df = (a + b*c) * (pod.probe * pod.object - pod.exit)

        pod.exit += df
        err_exit += np.mean(u.abs2(df))

    return err_fmag, err_exit


def _log_likelihood(diff_view):
    """
    Calculates the log-likelihood for a diffraction view.

    Parameters
    ----------
    diff_view : View
        View to diffraction data

    Returns
    -------
    ll_error :  float
        Log-likelihood error
    """
    I = diff_view.data
    LL = np.zeros_like(I)
    for name, pod in diff_view.pods.items():
        LL += pod.downsample(u.abs2(pod.fw(pod.probe * pod.object)))
    return np.sum(diff_view.pod.mask * (LL - I)**2 / (I + 1.)) / np.prod(LL.shape)


@register()
class DM_scanning_mirror(projectional.DM):
    """
    Modified version of the DM engine. The modification applies a phase
    gradient before propagation to account for changes in the beam angle during
    the scan.
    """

    SUPPORTED_MODELS = [ScanningMirrorModel]

    def __init__(self, ptycho_parent, pars=None):
        super(DM_scanning_mirror, self).__init__(ptycho_parent, pars)

    def fourier_update(self):
        """
        DM Fourier constraint update (including DM step).
        """
        error_dct = {}
        for name, di_view in self.di.views.items():
            if not di_view.active:
                continue
            pbound = self.pbound_scan[di_view.storage.label]
            err_fmag, err_exit = _projection_update_generalized(di_view,
                                                               self._a,
                                                               self._b,
                                                               self._c, pbound)
            if self.p.compute_log_likelihood:
                err_phot = _log_likelihood(di_view)
            else:
                err_phot = 0.
            error_dct[name] = np.array([err_fmag, err_phot, err_exit])

        return error_dct

    def engine_initialize(self):
        super(DM_scanning_mirror, self).engine_initialize()
        # Overwrite position refinement engine
        self.position_refinement = ScanningMirrorGridSearch(
            self.p.position_refinement,
            self.ob,
            metric=self.p.position_refinement.metric
        )
        if self.p.position_refinement.stop is None:
            self.p.position_refinement.stop = self.p.numiter
        if self.p.position_refinement.start is None:
            self.p.position_refinement.start = 0


class ScanningMirrorGridSearch(GridSearchRefine):
    """
    GridSearch position refinement engine that works with scanning mirror data.
    So far, that only means that it passes shift arguments to the propagator.
    No refinemnt of shifts is implemented.
    """

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
            af2 += pod.downsample(u.abs2(pod.fw(pod.probe * obj)))
        return np.sum(di_view.pod.mask * (np.sqrt(af2) - np.sqrt(
            np.abs(di_view.data))) ** 2) / di_view.pod.mask.sum()

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
            af2 += pod.downsample(u.abs2(pod.fw(pod.probe * obj)))
        return (np.sum(di_view.pod.mask * (af2 - di_view.data) ** 2 / (
                    di_view.data + 1.)) / np.prod(af2.shape))

