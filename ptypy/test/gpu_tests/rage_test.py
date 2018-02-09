'''
What the hell is the difference between these two things!?!?!?
'''


from ptypy import utils as u
import utils as tu
from ptypy.gpu import constraints as con
from ptypy.gpu import data_utils as du
from ptypy.gpu import array_utils as au
from ptypy.gpu.error_metrics import  far_field_error, realspace_error
from ptypy.gpu import object_probe_interaction as opi
import ptypy.gpu.propagation as prop
import numpy as np

def get_numpy_constrained_pods(PodPtychoInstance):
    out = []
    error_dct = {}
    f = {}
    for name, diff_view in PodPtychoInstance.di.views.iteritems():
        if not diff_view.active:
            continue

        # Buffer for accumulated photons
        af2 = np.zeros_like(diff_view.data)

        # Get measured data
        I = diff_view.data

        # Get the mask
        fmask = diff_view.pod.mask

        # Propagate the exit waves
        for name, pod in diff_view.pods.iteritems():
            if not pod.active:
                continue
            # f.append((1 + alpha) * pod.probe * pod.object - alpha * pod.exit)
            constrained = (1 + alpha) * pod.probe * pod.object - alpha * pod.exit
            f[name] = pod.fw(constrained)
            af2 += u.abs2(f[name])

        fmag = np.sqrt(np.abs(I))

        af = np.sqrt(af2)
        fdev = af - fmag
        err_fmag = np.sum(fmask * fdev ** 2) / fmask.sum()
        err_exit = 0.0
        if pbound is None:
            # No power bound
            fm = (1 - fmask) + fmask * fmag / (af + 1e-10)
            for name, pod in diff_view.pods.iteritems():
                if not pod.active:
                    continue
                df = pod.bw(fm * f[name]) - pod.probe * pod.object
                pod.exit += df
                err_exit += np.mean(u.abs2(df))
        elif err_fmag > pbound:
            # Power bound is applied
            renorm = np.sqrt(pbound / err_fmag)
            fm = (1 - fmask) + fmask * (fmag + fdev * renorm) / (af + 1e-10)
            for name, pod in diff_view.pods.iteritems():
                if not pod.active:
                    continue
                df = pod.bw(fm * f[name]) - pod.probe * pod.object
                pod.exit += df
                err_exit += np.mean(u.abs2(df))
        else:
            # Within power bound so no constraint applied.
            for name, pod in diff_view.pods.iteritems():
                if not pod.active:
                    continue
                df = alpha * (pod.probe * pod.object - pod.exit)
                pod.exit += df
                err_exit += np.mean(u.abs2(df))




        if pbound is not None:
            # rescale the fmagnitude error to some meaning !!!
            # PT: I am not sure I agree with this.
            err_fmag /= pbound
        out.append(err_exit)
    return np.array(out)


PodPtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
VecPtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
vectorised_scan = du.pod_to_arrays(VecPtychoInstance, 'S0000')

addr = vectorised_scan['meta']['addr']  # probably want to extract these at a later date, but just to get stuff going...
probe = vectorised_scan['probe']
obj = vectorised_scan['obj']
Idata = vectorised_scan['diffraction']
mask = vectorised_scan['mask']
exit_wave = vectorised_scan['exit wave']
first_view_id = vectorised_scan['meta']['view_IDs'][0]
propagator = VecPtychoInstance.di.V[first_view_id].pod.geometry.propagator
alpha = 1.0

constrained = con.difference_map_realspace_constraint(obj, probe, exit_wave, addr, alpha)
f = prop.farfield_propagator(constrained, prefilter=propagator.pre_fft, postfilter=propagator.post_fft, direction='forward')
pa, oa, ea, da, ma = zip(*addr[:, 0])
af2 = au.sum_to_buffer(au.abs2(f), Idata.shape, ea, da)
fmag = np.sqrt(Idata)
af = np.sqrt(af2)


err_fmag = far_field_error(af, fmag, mask) # need to check how this is devectorised

probe_object = opi.scan_and_multiply(probe, obj, exit_wave.shape, addr[:, 0])


fm = np.zeros(shape=exit_wave.shape, dtype=np.float64)
for _pa, _oa, ea, da, ma in addr[:, 0]:
    fm[ea[0]] = (1 - mask[ma[0]]) + mask[ma[0]] * fmag[da[0]] / (af[da[0]] + 1e-10)

normalised_fmag = np.multiply(fm, f)


backpropagated_solution = prop.farfield_propagator(normalised_fmag,
                                                   propagator.post_fft.conj(),
                                                   propagator.pre_fft.conj(),
                                                   direction='backward')
# #
df = np.subtract(backpropagated_solution, probe_object)

err_exit = realspace_error(df)

ptypy_fm = get_numpy_constrained_pods(PodPtychoInstance)

print err_exit.dtype, ptypy_fm.dtype

np.testing.assert_allclose(err_exit, ptypy_fm)

# import pylab as plt
# plt.close('all')
# plt.figure(1)
# # plt.plot(x, np.abs(f).sum(1).sum(1), x, np.abs(f_pods).sum(1).sum(1))
# plt.imshow((np.abs(f[1])/np.abs(f_pods[1])), vmin=0.9, vmax=1.1)
# plt.colorbar()
# plt.show()


#




def basic_fourier_update(diff_view, pbound=None, alpha=1., LL_error=True):
    """\
    Fourier update a single view using its associated pods.
    Updates on all pods' exit waves.

    Parameters
    ----------
    diff_view : View
        View to diffraction data

    alpha : float, optional
        Mixing between old and new exit wave. Valid interval ``[0, 1]``

    pbound : float, optional
        Power bound. Fourier update is bypassed if the quadratic deviation
        between diffraction data and `diff_view` is below this value.
        If ``None``, fourier update always happens.

    LL_error : bool
        If ``True``, calculates log-likelihood and puts it in the last entry
        of the returned error vector, else puts in ``0.0``

    Returns
    -------
    error : ndarray
        1d array, ``error = np.array([err_fmag, err_phot, err_exit])``.

        - `err_fmag`, Fourier magnitude error; quadratic deviation from
          root of experimental data
        - `err_phot`, quadratic deviation from experimental data (photons)
        - `err_exit`, quadratic deviation of exit waves before and after
          Fourier iteration
    """
    # Prepare dict for storing propagated waves
    f = {}

    # Buffer for accumulated photons
    af2 = np.zeros_like(diff_view.data)

    # Get measured data
    I = diff_view.data

    # Get the mask
    fmask = diff_view.pod.mask

    # Propagate the exit waves
    for name, pod in diff_view.pods.iteritems():
        if not pod.active:
            continue

        f[name] = pod.fw((1 + alpha) * pod.probe * pod.object
                         - alpha * pod.exit)



        af2 += u.abs2(f[name])

    fmag = np.sqrt(np.abs(I))
    af = np.sqrt(af2)

    # Fourier magnitudes deviations
    fdev = af - fmag
    err_fmag = np.sum(fmask * fdev**2) / fmask.sum()
    err_exit = 0.

    if pbound is None:
        # No power bound
        fm = (1 - fmask) + fmask * fmag / (af + 1e-10)
        for name, pod in diff_view.pods.iteritems():
            if not pod.active:
                continue
            df = pod.bw(fm * f[name]) - pod.probe * pod.object
            pod.exit += df
            err_exit += np.mean(u.abs2(df))
    elif err_fmag > pbound:
        # Power bound is applied
        renorm = np.sqrt(pbound / err_fmag)
        fm = (1 - fmask) + fmask * (fmag + fdev * renorm) / (af + 1e-10)
        for name, pod in diff_view.pods.iteritems():
            if not pod.active:
                continue
            df = pod.bw(fm * f[name]) - pod.probe * pod.object
            pod.exit += df
            err_exit += np.mean(u.abs2(df))
    else:
        # Within power bound so no constraint applied.
        for name, pod in diff_view.pods.iteritems():
            if not pod.active:
                continue
            df = alpha * (pod.probe * pod.object - pod.exit)
            pod.exit += df
            err_exit += np.mean(u.abs2(df))

    if pbound is not None:
        # rescale the fmagnitude error to some meaning !!!
        # PT: I am not sure I agree with this.
        err_fmag /= pbound

    return np.array([err_fmag, err_phot, err_exit])


import numpy as np

from ptypy.gpu.error_metrics import log_likelihood, far_field_error, realspace_error
from ptypy.gpu.object_probe_interaction import difference_map_realspace_constraint, scan_and_multiply
from ptypy.gpu.propagation import farfield_propagator
import ptypy.gpu.array_utils as au



def difference_map_fourier_constraint(mask, Idata, obj, probe, exit_wave, addr, prefilter, postfilter, pbound=None, alpha=1.0, LL_error=True):
    '''
    This kernel just performs the fourier renormalisation.
    :param mask. The nd mask array
    :param diffraction. The nd diffraction data
    :param farfield_stack. The current iterant.
    :param addr. The addresses of the stacks.
    :return: The updated iterant
            : fourier errors
    '''
    view_dlayer = 0 # what is this?
    addr_info = addr[:,(view_dlayer)] # addresses, object references
    # Buffer for accumulated photons
    # For log likelihood error # need to double check this adp
    if LL_error is True:
        err_phot = log_likelihood(probe, obj, mask, exit_wave, Idata, prefilter, postfilter, addr)
    else:
        err_phot = np.zeros(Idata.shape[0])

    # # Propagate the exit waves
    constrained = difference_map_realspace_constraint(obj, probe, exit_wave, addr, alpha)
    f = farfield_propagator(constrained, prefilter, postfilter, direction='forward')

    pa, oa, ea, da, ma = zip(*addr_info)
    af2 = au.sum_to_buffer(au.abs2(f), Idata.shape, ea, da)

    fmag = np.sqrt(np.abs(Idata))
    af = np.sqrt(af2)
    # # Fourier magnitudes deviations

    err_fmag = far_field_error(af, fmag, mask)

    probe_object = scan_and_multiply(probe, obj, exit_wave.shape, addr_info)

    fm = None

    for _pa, _oa, ea, da, ma in addr_info:
        if (pbound is None) or (err_fmag[da[0]] > pbound[da[0]]):
            # No power bound
            if pbound is None:
                fm = np.zeros_like(exit_wave)
                fm[ea[0]] = (1 - mask[ma[0]]) + mask[ma[0]] * fmag[da[0]] / (af[da[0]] + 1e-10)
            elif err_fmag[da[0]] > pbound[da[0]]:
                # Power bound is applied
                fdev = af[da[0]] - fmag[da[0]]
                fm = np.zeros_like(exit_wave)
                fm[ea[0]] = (1 - mask[ma[0]]) + mask[ma[0]] * (fmag[da[0]] + fdev[da[0]] * np.sqrt(pbound[da[0]] / err_fmag[da[0]])) / (af[da[0]] + 1e-10)


    if fm is None:
        df = np.multiply(alpha, (np.subtract(probe_object, exit_wave)))
    else:
        backpropagated_solution = farfield_propagator(np.multiply(fm, f),
                                                      prefilter.conj(),
                                                      postfilter.conj(),
                                                      direction='backward')

        df = np.subtract(backpropagated_solution, probe_object)

    exit_wave += df
    err_exit = realspace_error(df)

    if pbound is not None:
        # rescale the fmagnitude error to some meaning !!!
        # PT: I am not sure I agree with this.
        err_fmag /= pbound
    #
    return exit_wave, np.array([err_fmag, err_phot, err_exit])
