'''
The tests for the constraints
'''

import unittest
from . import utils as tu
import numpy as np
from copy import deepcopy

from ptypy.accelerate.array_based import constraints as con
from ptypy.accelerate.array_based import data_utils as du
from ptypy.accelerate.array_based.constraints import difference_map_fourier_constraint, renormalise_fourier_magnitudes, get_difference
from archive.array_based.error_metrics import far_field_error
from ptypy.accelerate.array_based.object_probe_interaction import difference_map_realspace_constraint, scan_and_multiply
from archive.array_based.propagation import farfield_propagator
import ptypy.accelerate.array_based.array_utils as au
from ptypy.accelerate.array_based import COMPLEX_TYPE, FLOAT_TYPE

from . import have_cuda, only_if_cuda_available
if have_cuda():
    from archive.cuda_extension.accelerate.cuda import constraints as gcon
    from archive.cuda_extension.accelerate.cuda import get_difference as gget_difference
    from archive.cuda_extension.accelerate.cuda import renormalise_fourier_magnitudes as grenormalise_fourier_magnitudes
    from archive.cuda_extension.accelerate.cuda import difference_map_fourier_constraint as gdifference_map_fourier_constraint
    from archive.cuda_extension.accelerate.cuda.config import init_gpus, reset_function_cache
    init_gpus(0)

@only_if_cuda_available
class ConstraintsTest(unittest.TestCase):

    def tearDown(self):
        # reset the cached GPU functions after each test
        reset_function_cache()

    def test_get_difference_UNITY(self):
        alpha = 1.0
        pbound = None
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        master_pod = PtychoInstance.diff.V[first_view_id].pod
        propagator = master_pod.geometry.propagator
        addr_info = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        Idata = vectorised_scan['diffraction']
        mask = vectorised_scan['mask']

        # # Propagate the exit waves
        probe_object = scan_and_multiply(probe, obj, exit_wave.shape, addr_info)
        constrained = difference_map_realspace_constraint(probe_object, exit_wave, alpha)
        f = farfield_propagator(constrained, propagator.pre_fft, propagator.post_fft, direction='forward')
        pa, oa, ea, da, ma = zip(*addr_info)
        af2 = au.sum_to_buffer(au.abs2(f), Idata.shape, ea, da, dtype=FLOAT_TYPE)

        fmag = np.sqrt(np.abs(Idata))
        af = np.sqrt(af2)
        # # Fourier magnitudes deviations(current_solution, pbound, measured_solution, mask, addr)
        err_fmag = far_field_error(af, fmag, mask)
        renormed_f = renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound)


        backpropagated_solution = farfield_propagator(renormed_f,
                                                      propagator.post_fft.conj(),
                                                      propagator.pre_fft.conj(),
                                                      direction='backward')

        difference = get_difference(addr_info, alpha, backpropagated_solution, err_fmag, exit_wave, pbound, probe_object)
        gdifference = gget_difference(addr_info, alpha, backpropagated_solution, err_fmag, exit_wave, pbound, probe_object)

        np.testing.assert_allclose(difference, gdifference, rtol=6e-5)
        # Detailed errors
        max_relerr = np.max(np.abs((gdifference-difference) / difference))
        max_abserr = np.max(np.abs(gdifference-difference))
        print("Max errors: rel={}, abs={}".format(max_relerr, max_abserr))

    def test_get_difference_pbound_UNITY(self):
        alpha = 1.0
        pbound = 0.597053604126
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        master_pod = PtychoInstance.diff.V[first_view_id].pod
        propagator = master_pod.geometry.propagator
        addr_info = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        Idata = vectorised_scan['diffraction']
        mask = vectorised_scan['mask']

        # # Propagate the exit waves
        probe_object = scan_and_multiply(probe, obj, exit_wave.shape, addr_info)
        constrained = difference_map_realspace_constraint(probe_object, exit_wave, alpha)
        f = farfield_propagator(constrained, propagator.pre_fft, propagator.post_fft, direction='forward')
        pa, oa, ea, da, ma = zip(*addr_info)
        af2 = au.sum_to_buffer(au.abs2(f), Idata.shape, ea, da, dtype=FLOAT_TYPE)

        fmag = np.sqrt(np.abs(Idata))
        af = np.sqrt(af2)
        # # Fourier magnitudes deviations(current_solution, pbound, measured_solution, mask, addr)
        err_fmag = far_field_error(af, fmag, mask)
        err_fmag = np.ones_like(err_fmag) * 145.824958919
        renormed_f = renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound)

        backpropagated_solution = farfield_propagator(renormed_f,
                                                      propagator.post_fft.conj(),
                                                      propagator.pre_fft.conj(),
                                                      direction='backward')

        difference = get_difference(addr_info, alpha, backpropagated_solution, err_fmag, exit_wave, pbound, probe_object)
        gdifference = gget_difference(addr_info, alpha, backpropagated_solution, err_fmag, exit_wave, pbound, probe_object)

        np.testing.assert_allclose(difference, gdifference, rtol=6e-5)
        # Detailed errors
        max_relerr = np.max(np.abs((gdifference-difference) / difference))
        max_abserr = np.max(np.abs(gdifference-difference))
        print("Max errors: rel={}, abs={}".format(max_relerr, max_abserr))

    def test_get_difference_no_update_UNITY(self):
        alpha = 1.0
        pbound = 0.597053604126
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        master_pod = PtychoInstance.diff.V[first_view_id].pod
        propagator = master_pod.geometry.propagator
        addr_info = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        Idata = vectorised_scan['diffraction']
        mask = vectorised_scan['mask']

        # # Propagate the exit waves
        probe_object = scan_and_multiply(probe, obj, exit_wave.shape, addr_info)
        constrained = difference_map_realspace_constraint(probe_object, exit_wave, alpha)
        f = farfield_propagator(constrained, propagator.pre_fft, propagator.post_fft, direction='forward')
        pa, oa, ea, da, ma = zip(*addr_info)
        af2 = au.sum_to_buffer(au.abs2(f), Idata.shape, ea, da, dtype=FLOAT_TYPE)

        fmag = np.sqrt(np.abs(Idata))
        af = np.sqrt(af2)
        # # Fourier magnitudes deviations(current_solution, pbound, measured_solution, mask, addr)
        err_fmag = far_field_error(af, fmag, mask)
        err_fmag = np.ones_like(err_fmag) * 0.4
        renormed_f = renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound)

        backpropagated_solution = farfield_propagator(renormed_f,
                                                      propagator.post_fft.conj(),
                                                      propagator.pre_fft.conj(),
                                                      direction='backward')

        difference = get_difference(addr_info, alpha, backpropagated_solution, err_fmag, exit_wave, pbound, probe_object)
        gdifference = gget_difference(addr_info, alpha, backpropagated_solution, err_fmag, exit_wave, pbound, probe_object)

        # result is actually all-zero, so array_equal works
        np.testing.assert_array_equal(difference, gdifference)

    def test_renormalise_fourier_magnitudes_UNITY(self):
        alpha = 1.0
        pbound = None
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        master_pod = PtychoInstance.diff.V[first_view_id].pod
        propagator = master_pod.geometry.propagator
        addr_info = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        Idata = vectorised_scan['diffraction']
        mask = vectorised_scan['mask']
        # # Propagate the exit waves
        probe_object = scan_and_multiply(probe, obj, exit_wave.shape, addr_info)
        constrained = difference_map_realspace_constraint(probe_object, exit_wave, alpha)
        f = farfield_propagator(constrained, propagator.pre_fft, propagator.post_fft, direction='forward')
        pa, oa, ea, da, ma = zip(*addr_info)
        af2 = au.sum_to_buffer(au.abs2(f), Idata.shape, ea, da, dtype=FLOAT_TYPE)

        fmag = np.sqrt(np.abs(Idata))
        af = np.sqrt(af2)
        # # Fourier magnitudes deviations(current_solution, pbound, measured_solution, mask, addr)
        err_fmag = far_field_error(af, fmag, mask)

        renormed_f = renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound)
        grenormed_f = grenormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound)

        np.testing.assert_allclose(renormed_f, grenormed_f, rtol=1e-6)

    def test_renormalise_fourier_magnitudes_pbound_UNITY(self):
        alpha = 1.0
        pbound = 0.597053604126
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        master_pod = PtychoInstance.diff.V[first_view_id].pod
        propagator = master_pod.geometry.propagator
        addr_info = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        Idata = vectorised_scan['diffraction']
        mask = vectorised_scan['mask']

        # # Propagate the exit waves
        probe_object = scan_and_multiply(probe, obj, exit_wave.shape, addr_info)
        constrained = difference_map_realspace_constraint(probe_object, exit_wave, alpha)
        f = farfield_propagator(constrained, propagator.pre_fft, propagator.post_fft, direction='forward')
        pa, oa, ea, da, ma = zip(*addr_info)
        af2 = au.sum_to_buffer(au.abs2(f), Idata.shape, ea, da, dtype=FLOAT_TYPE)

        fmag = np.sqrt(np.abs(Idata))
        af = np.sqrt(af2)
        # # Fourier magnitudes deviations(current_solution, pbound, measured_solution, mask, addr)
        err_fmag = far_field_error(af, fmag, mask)
        err_fmag = np.ones_like(err_fmag) * 145.824958919
        renormed_f = renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound)
        grenormed_f = grenormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound)

        np.testing.assert_allclose(renormed_f, grenormed_f, rtol=1e-6)

    def test_renormalise_fourier_magnitudes_no_update_UNITY(self):
        alpha = 1.0
        pbound = 0.597053604126
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        master_pod = PtychoInstance.diff.V[first_view_id].pod
        propagator = master_pod.geometry.propagator
        addr_info = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        Idata = vectorised_scan['diffraction']
        mask = vectorised_scan['mask']

        # # Propagate the exit waves
        probe_object = scan_and_multiply(probe, obj, exit_wave.shape, addr_info)
        constrained = difference_map_realspace_constraint(probe_object, exit_wave, alpha)
        f = farfield_propagator(constrained, propagator.pre_fft, propagator.post_fft, direction='forward')
        pa, oa, ea, da, ma = zip(*addr_info)
        af2 = au.sum_to_buffer(au.abs2(f), Idata.shape, ea, da, dtype=FLOAT_TYPE)

        fmag = np.sqrt(np.abs(Idata))
        af = np.sqrt(af2)
        # # Fourier magnitudes deviations(current_solution, pbound, measured_solution, mask, addr)
        err_fmag = far_field_error(af, fmag, mask)
        err_fmag = np.ones_like(err_fmag) * 0.4
        renormed_f = renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound)
        grenormed_f = grenormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound)

        np.testing.assert_array_equal(renormed_f, grenormed_f)

    def test_difference_map_fourier_constraint_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        master_pod = PtychoInstance.diff.V[first_view_id].pod
        propagator = master_pod.geometry.propagator

        # take a copy of the starting point, as function updates in-place
        exit_wave_start = np.copy(vectorised_scan['exit wave'])

        errors = difference_map_fourier_constraint(
            vectorised_scan['mask'],
            vectorised_scan['diffraction'],
            vectorised_scan['obj'],
            vectorised_scan['probe'],
            vectorised_scan['exit wave'],
            vectorised_scan['meta']['addr'],
            prefilter=propagator.pre_fft,
            postfilter=propagator.post_fft,
            pbound=None,
            alpha=1.0,
            LL_error=True)

        # keep result, copy original back
        exit_wave = vectorised_scan['exit wave']
        vectorised_scan['exit wave'] = exit_wave_start

        gerrors = gdifference_map_fourier_constraint(vectorised_scan['mask'],
                                                    vectorised_scan['diffraction'],
                                                    vectorised_scan['obj'],
                                                    vectorised_scan['probe'],
                                                    vectorised_scan['exit wave'],
                                                    vectorised_scan['meta']['addr'],
                                                    prefilter=propagator.pre_fft,
                                                    postfilter=propagator.post_fft,
                                                    pbound=None,
                                                    alpha=1.0,
                                                    LL_error=True)

        gexit_wave = vectorised_scan['exit wave']

        max_relerr = np.max(np.abs((exit_wave-gexit_wave) / exit_wave))
        mean_relerr = np.mean(np.abs((exit_wave-gexit_wave) / exit_wave))
        max_abserr = np.max(np.abs(exit_wave-gexit_wave))
        mean_abserr = np.mean(np.abs(exit_wave-gexit_wave))
        print("Exit wave max errors: rel={}, abs={}".format(max_relerr, max_abserr))
        print("Exit wave mean errors: rel={}, abs={}".format(mean_relerr, mean_abserr))

        max_relerr = np.max(np.abs((errors-gerrors) / errors), axis=None)
        mean_relerr = np.mean(np.abs((errors-gerrors) / errors), axis=None)
        max_abserr = np.max(np.abs(errors-gerrors), axis=None)
        mean_abserr = np.mean(np.abs(errors-gerrors), axis=None)
        print("Errors max errors: rel={}, abs={}".format(max_relerr, max_abserr))
        print("Errors mean errors: rel={}, abs={}".format(mean_relerr, mean_abserr))

        np.testing.assert_allclose(exit_wave, gexit_wave, rtol=3e-1, atol=18)
        np.testing.assert_allclose(errors, gerrors, rtol=3e-4)

    def test_difference_map_fourier_constraint_pbound_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        master_pod = PtychoInstance.diff.V[first_view_id].pod
        propagator = master_pod.geometry.propagator
        pbound = 0.597053604126

         # take a copy of the starting point, as function updates in-place
        exit_wave_start = np.copy(vectorised_scan['exit wave'])

        errors = difference_map_fourier_constraint(vectorised_scan['mask'],
                                                              vectorised_scan['diffraction'],
                                                              vectorised_scan['obj'],
                                                              vectorised_scan['probe'],
                                                              vectorised_scan['exit wave'],
                                                              vectorised_scan['meta']['addr'],
                                                              prefilter=propagator.pre_fft,
                                                              postfilter=propagator.post_fft,
                                                              pbound=pbound,
                                                              alpha=1.0,
                                                              LL_error=True)

        # keep result, copy original back
        exit_wave = vectorised_scan['exit wave']
        vectorised_scan['exit wave'] = exit_wave_start

        gerrors = gdifference_map_fourier_constraint(vectorised_scan['mask'],
                                                              vectorised_scan['diffraction'],
                                                              vectorised_scan['obj'],
                                                              vectorised_scan['probe'],
                                                              vectorised_scan['exit wave'],
                                                              vectorised_scan['meta']['addr'],
                                                              prefilter=propagator.pre_fft,
                                                              postfilter=propagator.post_fft,
                                                              pbound=pbound,
                                                              alpha=1.0,
                                                              LL_error=True)

        gexit_wave = vectorised_scan['exit wave']

        max_relerr = np.max(np.abs((exit_wave-gexit_wave) / exit_wave))
        mean_relerr = np.mean(np.abs((exit_wave-gexit_wave) / exit_wave))
        max_abserr = np.max(np.abs(exit_wave-gexit_wave))
        mean_abserr = np.mean(np.abs(exit_wave-gexit_wave))
        print("Exit wave max errors: rel={}, abs={}".format(max_relerr, max_abserr))
        print("Exit wave mean errors: rel={}, abs={}".format(mean_relerr, mean_abserr))

        max_relerr = np.max(np.abs((errors-gerrors) / errors), axis=None)
        mean_relerr = np.mean(np.abs((errors-gerrors) / errors), axis=None)
        max_abserr = np.max(np.abs(errors-gerrors), axis=None)
        mean_abserr = np.mean(np.abs(errors-gerrors), axis=None)
        print("Errors max errors: rel={}, abs={}".format(max_relerr, max_abserr))
        print("Errors mean errors: rel={}, abs={}".format(mean_relerr, mean_abserr))

        np.testing.assert_allclose(exit_wave, gexit_wave, rtol=4e-1, atol=16)
        np.testing.assert_allclose(errors, gerrors, rtol=4e-4)

    def test_difference_map_fourier_constraint_no_update_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        master_pod = PtychoInstance.diff.V[first_view_id].pod
        propagator = master_pod.geometry.propagator
        pbound = 200.0

         # take a copy of the starting point, as function updates in-place
        exit_wave_start = deepcopy(vectorised_scan['exit wave'])

        errors = difference_map_fourier_constraint(vectorised_scan['mask'],
                                                              vectorised_scan['diffraction'],
                                                              vectorised_scan['obj'],
                                                              vectorised_scan['probe'],
                                                              vectorised_scan['exit wave'],
                                                              vectorised_scan['meta']['addr'],
                                                              prefilter=propagator.pre_fft,
                                                              postfilter=propagator.post_fft,
                                                              pbound=pbound,
                                                              alpha=1.0,
                                                              LL_error=True)
        # keep result, copy original back
        exit_wave = vectorised_scan['exit wave']
        vectorised_scan['exit wave'] = exit_wave_start


        gerrors = gdifference_map_fourier_constraint(vectorised_scan['mask'],
                                                              vectorised_scan['diffraction'],
                                                              vectorised_scan['obj'],
                                                              vectorised_scan['probe'],
                                                              vectorised_scan['exit wave'],
                                                              vectorised_scan['meta']['addr'],
                                                              prefilter=propagator.pre_fft,
                                                              postfilter=propagator.post_fft,
                                                              pbound=pbound,
                                                              alpha=1.0,
                                                              LL_error=True)

        gexit_wave = vectorised_scan['exit wave']

        max_relerr = np.nanmax(np.abs((exit_wave-gexit_wave) / exit_wave))
        mean_relerr = np.nanmean(np.abs((exit_wave-gexit_wave) / exit_wave))
        max_abserr = np.max(np.abs(exit_wave-gexit_wave))
        mean_abserr = np.mean(np.abs(exit_wave-gexit_wave))
        print("Exit wave max errors: rel={}, abs={}".format(max_relerr, max_abserr))
        print("Exit wave mean errors: rel={}, abs={}".format(mean_relerr, mean_abserr))

        max_relerr = np.nanmax(np.abs((errors-gerrors) / errors), axis=None)
        mean_relerr = np.nanmean(np.abs((errors-gerrors) / errors), axis=None)
        max_abserr = np.max(np.abs(errors-gerrors), axis=None)
        mean_abserr = np.mean(np.abs(errors-gerrors), axis=None)
        print("Errors max errors: rel={}, abs={}".format(max_relerr, max_abserr))
        print("Errors mean errors: rel={}, abs={}".format(mean_relerr, mean_abserr))

        np.testing.assert_allclose(exit_wave, gexit_wave, rtol=5e-1, atol=10)
        np.testing.assert_allclose(errors, gerrors, rtol=3e-4)

    def test_difference_map_fourier_constraint_pbound_is_none_with_realspace_error_and_LL_error(self):

        alpha = 1.0 # feedback constant
        pbound = None  # the power bound
        num_object_modes = 1 # for example
        num_probe_modes = 2 # for example

        N = 4 # number of measured points
        M = N * num_object_modes * num_probe_modes # exit wave length
        A = 2 # for example
        B = 4 # for example
        npts_greater_than = int(np.sqrt(N)) # object is bigger than the probe by this amount
        C = A + npts_greater_than
        D = B + npts_greater_than

        err_fmag = np.empty(shape=(N,), dtype=FLOAT_TYPE)# deviation from the diffraction pattern for each af
        exit_wave = np.empty(shape=(M, A, B), dtype=COMPLEX_TYPE) # exit wave
        addr_info = np.empty(shape=(M, 5, 3), dtype=np.int32)# the address book
        Idata = np.empty(shape=(N, A, B), dtype=FLOAT_TYPE)# the measured intensities NxAxB
        mask = np.empty(shape=(N, A, B), dtype=np.int32)# the masks for the measured magnitudes either 1xAxB or NxAxB
        probe = np.empty(shape=(num_probe_modes, A, B), dtype=COMPLEX_TYPE) # the probe function
        obj = np.empty(shape=(num_object_modes, C, D), dtype=COMPLEX_TYPE)  # the object function
        prefilter = np.empty(shape=(A, B), dtype=COMPLEX_TYPE)
        postfilter = np.empty(shape=(A, B), dtype=COMPLEX_TYPE)


        # now fill it with stuff. Data won't ever look like this except in type and usage!
        Idata_fill = np.arange(np.prod(Idata.shape)).reshape(Idata.shape).astype(Idata.dtype)
        Idata[:] = Idata_fill

        obj_fill = np.array([ix + 1j*(ix**2) for ix in range(np.prod(obj.shape))]).reshape((num_object_modes, C, D))
        obj[:] = obj_fill

        probe_fill = np.array([ix + 1j*ix for ix in range(10, 10+np.prod(probe.shape), 1)]).reshape((num_probe_modes, A, B))
        probe[:] = probe_fill

        prefilter.fill(30.0 + 2.0j)# this would actually vary
        postfilter.fill(20.0 + 3.0j)# this too
        err_fmag_fill = np.ones((N,))
        err_fmag[:] = err_fmag_fill  # this shouldn't be used as pbound is None

        exit_wave_fill =  np.array([ix**2 + 1j*ix for ix in range(20, 20+np.prod(exit_wave.shape), 1)]).reshape((M, A, B))
        exit_wave[:] = exit_wave_fill

        mask_fill = np.ones_like(mask)
        mask_fill[::2, ::2] = 0 # checkerboard for testing
        mask[:] = mask_fill

        pa = np.zeros((M, 3), dtype=np.int32)
        for idx in range(num_probe_modes):
            if idx>0:
                pa[::idx,0]=idx # multimodal could work like this, but this is not a concrete thing.


        X, Y = np.meshgrid(range(npts_greater_than), range(npts_greater_than)) # assume square scan grid. Again, not always true.
        oa = np.zeros((M, 3), dtype=np.int32)
        oa[:N, 1] = X.ravel()
        oa[N:, 1] = X.ravel()
        oa[:N, 2] = Y.ravel()
        oa[N:, 2] = Y.ravel()
        for idx in range(num_object_modes):
            if idx>0:
                oa[::idx,0]=idx # multimodal could work like this, but this is not a concrete thing (less likely for object)
        ea = np.array([np.array([ix, 0, 0]) for ix in range(M)])
        da = np.array([np.array([ix, 0, 0]) for ix in range(N)]*num_probe_modes*num_object_modes)
        ma = np.zeros((M, 3), dtype=np.int32)

        addr_info[:, 0, :] = pa
        addr_info[:, 1, :] = oa
        addr_info[:, 2, :] = ea
        addr_info[:, 3, :] = da
        addr_info[:, 4, :] = ma

        gexit_wave = deepcopy(exit_wave)

        errors = con.difference_map_fourier_constraint(mask, Idata, obj, probe, exit_wave, addr_info, prefilter, postfilter, pbound=pbound, alpha=alpha, LL_error=True, do_realspace_error=True)
        gerrors = gcon.difference_map_fourier_constraint(mask, Idata, obj, probe, gexit_wave, addr_info, prefilter,
                                                     postfilter, pbound=pbound, alpha=alpha, LL_error=True,
                                                     do_realspace_error=True)
        
        np.testing.assert_allclose(gerrors,
                                   errors,
                                   rtol=1e-6,
                                   err_msg="The returned errors are not consistent.")

        np.testing.assert_allclose(gexit_wave,
                                   exit_wave,
                                   rtol=1e-6,
                                   err_msg="The expected in-place update of the exit wave didn't work properly.")

    def test_difference_map_fourier_constraint_pbound_is_none_no_error(self):

        alpha = 1.0  # feedback constant
        pbound = None  # the power bound
        num_object_modes = 1  # for example
        num_probe_modes = 2  # for example

        N = 4  # number of measured points
        M = N * num_object_modes * num_probe_modes  # exit wave length
        A = 2  # for example
        B = 4  # for example
        npts_greater_than = int(np.sqrt(N))  # object is bigger than the probe by this amount
        C = A + npts_greater_than
        D = B + npts_greater_than

        err_fmag = np.empty(shape=(N,), dtype=FLOAT_TYPE)  # deviation from the diffraction pattern for each af
        exit_wave = np.empty(shape=(M, A, B), dtype=COMPLEX_TYPE)  # exit wave
        addr_info = np.empty(shape=(M, 5, 3), dtype=np.int32)  # the address book
        Idata = np.empty(shape=(N, A, B), dtype=FLOAT_TYPE)  # the measured intensities NxAxB
        mask = np.empty(shape=(N, A, B),
                        dtype=np.int32)  # the masks for the measured magnitudes either 1xAxB or NxAxB
        probe = np.empty(shape=(num_probe_modes, A, B), dtype=COMPLEX_TYPE)  # the probe function
        obj = np.empty(shape=(num_object_modes, C, D), dtype=COMPLEX_TYPE)  # the object function
        prefilter = np.empty(shape=(A, B), dtype=COMPLEX_TYPE)
        postfilter = np.empty(shape=(A, B), dtype=COMPLEX_TYPE)

        # now fill it with stuff. Data won't ever look like this except in type and usage!
        Idata_fill = np.arange(np.prod(Idata.shape)).reshape(Idata.shape).astype(Idata.dtype)
        Idata[:] = Idata_fill

        obj_fill = np.array([ix + 1j * (ix ** 2) for ix in range(np.prod(obj.shape))]).reshape(
            (num_object_modes, C, D))
        obj[:] = obj_fill

        probe_fill = np.array([ix + 1j * ix for ix in range(10, 10 + np.prod(probe.shape), 1)]).reshape(
            (num_probe_modes, A, B))
        probe[:] = probe_fill

        prefilter.fill(30.0 + 2.0j)  # this would actually vary
        postfilter.fill(20.0 + 3.0j)  # this too
        err_fmag_fill = np.ones((N,))
        err_fmag[:] = err_fmag_fill  # this shouldn't be used as pbound is None

        exit_wave_fill = np.array(
            [ix ** 2 + 1j * ix for ix in range(20, 20 + np.prod(exit_wave.shape), 1)]).reshape((M, A, B))
        exit_wave[:] = exit_wave_fill

        mask_fill = np.ones_like(mask)
        mask_fill[::2, ::2] = 0  # checkerboard for testing
        mask[:] = mask_fill

        pa = np.zeros((M, 3), dtype=np.int32)
        for idx in range(num_probe_modes):
            if idx > 0:
                pa[::idx, 0] = idx  # multimodal could work like this, but this is not a concrete thing.

        X, Y = np.meshgrid(range(npts_greater_than),
                           range(npts_greater_than))  # assume square scan grid. Again, not always true.
        oa = np.zeros((M, 3), dtype=np.int32)
        oa[:N, 1] = X.ravel()
        oa[N:, 1] = X.ravel()
        oa[:N, 2] = Y.ravel()
        oa[N:, 2] = Y.ravel()
        for idx in range(num_object_modes):
            if idx > 0:
                oa[::idx,
                0] = idx  # multimodal could work like this, but this is not a concrete thing (less likely for object)
        ea = np.array([np.array([ix, 0, 0]) for ix in range(M)])
        da = np.array([np.array([ix, 0, 0]) for ix in range(N)] * num_probe_modes * num_object_modes)
        ma = np.zeros((M, 3), dtype=np.int32)

        addr_info[:, 0, :] = pa
        addr_info[:, 1, :] = oa
        addr_info[:, 2, :] = ea
        addr_info[:, 3, :] = da
        addr_info[:, 4, :] = ma

        gexit_wave = deepcopy(exit_wave)

        gerrors = gcon.difference_map_fourier_constraint(mask, Idata, obj, probe, gexit_wave, addr_info, prefilter,
                                                    postfilter, pbound=pbound, alpha=alpha, LL_error=False,
                                                    do_realspace_error=False)
        errors = con.difference_map_fourier_constraint(mask, Idata, obj, probe, exit_wave, addr_info, prefilter,
                                                    postfilter, pbound=pbound, alpha=alpha, LL_error=False,
                                                    do_realspace_error=False)
        
        np.testing.assert_allclose(gerrors,
                                   errors,
                                   rtol=1e-6,
                                   err_msg="The returned errors are not consistent.")

        np.testing.assert_allclose(gexit_wave,
                                   exit_wave,
                                   rtol=1e-6,
                                   err_msg="The expected in-place update of the exit wave didn't work properly.")


    def test_difference_map_fourier_constraint_pbound_is_not_none_with_realspace_and_LL_error(self):
        '''
        mixture of high and low p bound values respect to the fourier error
        '''
        #expected_fourier_error = np.array([6.07364329e+12, 8.87756439e+13, 9.12644403e+12, 1.39851186e+14])
        pbound = 8.86e13 # this should now mean some of the arrays update differently through the logic
        alpha = 1.0  # feedback constant
        num_object_modes = 1  # for example
        num_probe_modes = 2  # for example

        N = 4  # number of measured points
        M = N * num_object_modes * num_probe_modes  # exit wave length
        A = 2  # for example
        B = 4  # for example
        npts_greater_than = int(np.sqrt(N))  # object is bigger than the probe by this amount
        C = A + npts_greater_than
        D = B + npts_greater_than

        err_fmag = np.empty(shape=(N,), dtype=FLOAT_TYPE)  # deviation from the diffraction pattern for each af
        exit_wave = np.empty(shape=(M, A, B), dtype=COMPLEX_TYPE)  # exit wave
        addr_info = np.empty(shape=(M, 5, 3), dtype=np.int32)  # the address book
        Idata = np.empty(shape=(N, A, B), dtype=FLOAT_TYPE)  # the measured intensities NxAxB
        mask = np.empty(shape=(N, A, B),
                        dtype=np.int32)  # the masks for the measured magnitudes either 1xAxB or NxAxB
        probe = np.empty(shape=(num_probe_modes, A, B), dtype=COMPLEX_TYPE)  # the probe function
        obj = np.empty(shape=(num_object_modes, C, D), dtype=COMPLEX_TYPE)  # the object function
        prefilter = np.empty(shape=(A, B), dtype=COMPLEX_TYPE)
        postfilter = np.empty(shape=(A, B), dtype=COMPLEX_TYPE)

        # now fill it with stuff. Data won't ever look like this except in type and usage!
        Idata_fill = np.arange(np.prod(Idata.shape)).reshape(Idata.shape).astype(Idata.dtype)
        Idata[:] = Idata_fill

        obj_fill = np.array([ix + 1j * (ix ** 2) for ix in range(np.prod(obj.shape))]).reshape(
            (num_object_modes, C, D))
        obj[:] = obj_fill

        probe_fill = np.array([ix + 1j * ix for ix in range(10, 10 + np.prod(probe.shape), 1)]).reshape(
            (num_probe_modes, A, B))
        probe[:] = probe_fill

        prefilter.fill(30.0 + 2.0j)  # this would actually vary
        postfilter.fill(20.0 + 3.0j)  # this too
        err_fmag_fill = np.ones((N,))
        err_fmag[:] = err_fmag_fill  # this shouldn't be used as pbound is None

        exit_wave_fill = np.array(
            [ix ** 2 + 1j * ix for ix in range(20, 20 + np.prod(exit_wave.shape), 1)]).reshape((M, A, B))
        exit_wave[:] = exit_wave_fill

        mask_fill = np.ones_like(mask)
        mask_fill[::2, ::2] = 0  # checkerboard for testing
        mask[:] = mask_fill

        pa = np.zeros((M, 3), dtype=np.int32)
        for idx in range(num_probe_modes):
            if idx > 0:
                pa[::idx, 0] = idx  # multimodal could work like this, but this is not a concrete thing.

        X, Y = np.meshgrid(range(npts_greater_than),
                           range(npts_greater_than))  # assume square scan grid. Again, not always true.
        oa = np.zeros((M, 3), dtype=np.int32)
        oa[:N, 1] = X.ravel()
        oa[N:, 1] = X.ravel()
        oa[:N, 2] = Y.ravel()
        oa[N:, 2] = Y.ravel()
        for idx in range(num_object_modes):
            if idx > 0:
                oa[::idx,
                0] = idx  # multimodal could work like this, but this is not a concrete thing (less likely for object)
        ea = np.array([np.array([ix, 0, 0]) for ix in range(M)])
        da = np.array([np.array([ix, 0, 0]) for ix in range(N)] * num_probe_modes * num_object_modes)
        ma = np.zeros((M, 3), dtype=np.int32)

        addr_info[:, 0, :] = pa
        addr_info[:, 1, :] = oa
        addr_info[:, 2, :] = ea
        addr_info[:, 3, :] = da
        addr_info[:, 4, :] = ma

        gexit_wave = deepcopy(exit_wave)

        gerrors = gcon.difference_map_fourier_constraint(mask, Idata, obj, probe, gexit_wave, addr_info, prefilter,
                                                    postfilter, pbound=pbound, alpha=alpha, LL_error=True,
                                                    do_realspace_error=True)
        errors = con.difference_map_fourier_constraint(mask, Idata, obj, probe, exit_wave, addr_info, prefilter,
                                                    postfilter, pbound=pbound, alpha=alpha, LL_error=True,
                                                    do_realspace_error=True)

        np.testing.assert_allclose(gerrors,
                                   errors,
                                   rtol=1e-6,
                                   err_msg="The returned errors are not consistent.")

        np.testing.assert_allclose(gexit_wave,
                                   exit_wave,
                                   rtol=1e-6,
                                   err_msg="The expected in-place update of the exit wave didn't work properly.")


    @unittest.skip("The test doesn't work, but moonflower sample shows that this is actually working ok")
    def test_difference_map_iterator_with_probe_update(self):
        '''
        This test, assumes the logic below this function works fine, and just does some iterations of difference map on
        some spoof data to check that the combination works.
        '''
        num_iter = 2

        pbound = 8.86e13 # this should now mean some of the arrays update differently through the logic
        alpha = 1.0  # feedback constant
        # diffraction frame size
        B = 2  # for example
        C = 4  # for example

        # probe dimensions
        D = 2  # for example
        E = B
        F = C

        scan_pts = 2 # a 2x2 grid
        N = scan_pts**2 # the number of measurement points in a scan
        npts_greater_than = int(np.sqrt(N))  # object is bigger than the probe by this amount

        # object dimensions
        G = 1  # for example
        H = B + npts_greater_than
        I = C + npts_greater_than

        A = scan_pts ** 2 * G * D # number of exit waves

        err_fmag = np.empty(shape=(N,), dtype=FLOAT_TYPE)  # deviation from the diffraction pattern for each af
        exit_wave = np.empty(shape=(A, B, C), dtype=COMPLEX_TYPE)  # exit wave
        addr_info = np.empty(shape=(A, 5, 3), dtype=np.int32)  # the address book
        diffraction = np.empty(shape=(N, B, C), dtype=FLOAT_TYPE)  # the measured intensities NxAxB
        mask = np.empty(shape=(N, B, C),
                        dtype=np.int32)  # the masks for the measured magnitudes either 1xAxB or NxAxB
        probe = np.empty(shape=(D, E, F), dtype=COMPLEX_TYPE)  # the probe function
        obj = np.empty(shape=(G, H, I), dtype=COMPLEX_TYPE)  # the object function
        prefilter = np.empty(shape=(B, C), dtype=COMPLEX_TYPE)
        postfilter = np.empty(shape=(B, C), dtype=COMPLEX_TYPE)

        # now fill it with stuff. Data won't ever look like this except in type and usage!
        diffraction_fill = np.arange(np.prod(diffraction.shape)).reshape(diffraction.shape).astype(diffraction.dtype)
        diffraction[:] = diffraction_fill

        obj_fill = np.array([ix + 1j * (ix ** 2) for ix in range(np.prod(obj.shape))]).reshape(
            (G, H, I))
        obj[:] = obj_fill

        probe_fill = np.array([ix + 1j * ix for ix in range(10, 10 + np.prod(probe.shape), 1)]).reshape(
            (D, B, C))
        probe[:] = probe_fill

        prefilter.fill(30.0 + 2.0j)  # this would actually vary
        postfilter.fill(20.0 + 3.0j)  # this too
        err_fmag_fill = np.ones((N,))
        err_fmag[:] = err_fmag_fill  # this shouldn't be used as pbound is None

        exit_wave_fill = np.array(
            [ix ** 2 + 1j * ix for ix in range(20, 20 + np.prod(exit_wave.shape), 1)]).reshape((A, B, C))
        exit_wave[:] = exit_wave_fill

        mask_fill = np.ones_like(mask)
        # mask_fill[::2, ::2] = 0  # checkerboard for testing
        mask[:] = mask_fill

        pa = np.zeros((A, 3), dtype=np.int32)
        pa[:N,0] = 0
        pa[N:, 0] = 1

        X, Y = np.meshgrid(range(npts_greater_than),
                           range(npts_greater_than))  # assume square scan grid. Again, not always true.
        oa = np.zeros((A, 3), dtype=np.int32)
        oa[:N, 1] = X.ravel()
        oa[N:, 1] = X.ravel()
        oa[:N, 2] = Y.ravel()
        oa[N:, 2] = Y.ravel()


        ea = np.array([np.array([ix, 0, 0]) for ix in range(A)])
        da = np.array([np.array([ix, 0, 0]) for ix in range(N)] * D * G)
        ma = np.zeros((A, 3), dtype=np.int32)

        addr_info[:, 0, :] = pa
        addr_info[:, 1, :] = oa
        addr_info[:, 2, :] = ea
        addr_info[:, 3, :] = da
        addr_info[:, 4, :] = ma

        obj_weights = np.empty(shape=(G,), dtype=FLOAT_TYPE)
        obj_weights[:] = np.linspace(-1, 1, G)

        probe_weights = np.empty(shape=(D,), dtype=FLOAT_TYPE)
        probe_weights[:] = np.linspace(-1, 1, D)

        cfact_object = np.empty_like(obj)
        for idx in range(G):
            cfact_object[idx] = np.ones((H, I)) * 10 * (idx + 1)

        cfact_probe = np.empty_like(probe)

        for idx in range(D):
            cfact_probe[idx] = np.ones((B, C)) * 5 * (idx + 1)

        gexit_wave = deepcopy(exit_wave)
        gprobe = deepcopy(probe)
        gobj = deepcopy(obj)

        errors = con.difference_map_iterator(diffraction=diffraction,
                                             obj=obj,
                                             object_weights=obj_weights,
                                             cfact_object=cfact_object,
                                             mask=mask,
                                             probe=probe,
                                             cfact_probe=cfact_probe,
                                             probe_support=None,
                                             probe_weights=probe_weights,
                                             exit_wave=exit_wave,
                                             addr=addr_info,
                                             pre_fft=prefilter,
                                             post_fft=postfilter,
                                             pbound=pbound,
                                             overlap_max_iterations=10,
                                             update_object_first=False,
                                             obj_smooth_std=None,
                                             overlap_converge_factor=1.4e-3,
                                             probe_center_tol=None,
                                             probe_update_start=1,
                                             alpha=alpha,
                                             clip_object=None,
                                             LL_error=True,
                                             num_iterations=num_iter)

        gerrors = gcon.difference_map_iterator(diffraction=diffraction,
                                             obj=gobj,
                                             object_weights=obj_weights,
                                             cfact_object=cfact_object,
                                             mask=mask,
                                             probe=gprobe,
                                             cfact_probe=cfact_probe,
                                             probe_support=None,
                                             probe_weights=probe_weights,
                                             exit_wave=gexit_wave,
                                             addr=addr_info,
                                             pre_fft=prefilter,
                                             post_fft=postfilter,
                                             pbound=pbound,
                                             overlap_max_iterations=10,
                                             update_object_first=False,
                                             obj_smooth_std=None,
                                             overlap_converge_factor=1.4e-3,
                                             probe_center_tol=None,
                                             probe_update_start=1,
                                             alpha=alpha,
                                             clip_object=None,
                                             LL_error=True,
                                             num_iterations=num_iter)

        np.testing.assert_allclose(gerrors,
                                   errors,
                                   rtol=1e-6,
                                   err_msg="The returned errors are not consistent.")

        np.testing.assert_allclose(gprobe,
                                  probe,
                                  rtol=1e-6,
                                  err_msg="The returned probes are not consistent.")

        np.testing.assert_allclose(gobj,
                                  obj,
                                  rtol=1e-6,
                                  err_msg="The returned objects are not consistent.")

        np.testing.assert_allclose(gexit_wave[3,:,:],
                                   exit_wave[3,:,:],
                                   rtol=1e-6,
                                   atol=1e-4,
                                   err_msg="The returned exit_waves are not consistent.")

    def test_difference_map_iterator_with_no_probe_update_and_object_update(self):
        '''
        This test, assumes the logic below this function works fine, and just does some iterations of difference map on
        some spoof data to check that the combination works.
        '''
        num_iter = 1

        pbound = 8.86e13 # this should now mean some of the arrays update differently through the logic
        alpha = 1.0  # feedback constant
        # diffraction frame size
        B = 2  # for example
        C = 4  # for example

        # probe dimensions
        D = 2  # for example
        E = B
        F = C

        scan_pts = 2 # a 2x2 grid
        N = scan_pts**2 # the number of measurement points in a scan
        npts_greater_than = int(np.sqrt(N))  # object is bigger than the probe by this amount

        # object dimensions
        G = 1  # for example
        H = B + npts_greater_than
        I = C + npts_greater_than

        A = scan_pts ** 2 * G * D # number of exit waves

        err_fmag = np.empty(shape=(N,), dtype=FLOAT_TYPE)  # deviation from the diffraction pattern for each af
        exit_wave = np.empty(shape=(A, B, C), dtype=COMPLEX_TYPE)  # exit wave
        addr_info = np.empty(shape=(A, 5, 3), dtype=np.int32)  # the address book
        diffraction = np.empty(shape=(N, B, C), dtype=FLOAT_TYPE)  # the measured intensities NxAxB
        mask = np.empty(shape=(N, B, C),
                        dtype=np.int32)  # the masks for the measured magnitudes either 1xAxB or NxAxB
        probe = np.empty(shape=(D, E, F), dtype=COMPLEX_TYPE)  # the probe function
        obj = np.empty(shape=(G, H, I), dtype=COMPLEX_TYPE)  # the object function
        prefilter = np.empty(shape=(B, C), dtype=COMPLEX_TYPE)
        postfilter = np.empty(shape=(B, C), dtype=COMPLEX_TYPE)

        # now fill it with stuff. Data won't ever look like this except in type and usage!
        diffraction_fill = np.arange(np.prod(diffraction.shape)).reshape(diffraction.shape).astype(diffraction.dtype)
        diffraction[:] = diffraction_fill

        obj_fill = np.array([ix + 1j * (ix ** 2) for ix in range(np.prod(obj.shape))]).reshape(
            (G, H, I))
        obj[:] = obj_fill

        probe_fill = np.array([ix + 1j * ix for ix in range(10, 10 + np.prod(probe.shape), 1)]).reshape(
            (D, B, C))
        probe[:] = probe_fill

        prefilter.fill(30.0 + 2.0j)  # this would actually vary
        postfilter.fill(20.0 + 3.0j)  # this too
        err_fmag_fill = np.ones((N,))
        err_fmag[:] = err_fmag_fill  # this shouldn't be used as pbound is None

        exit_wave_fill = np.array(
            [ix ** 2 + 1j * ix for ix in range(20, 20 + np.prod(exit_wave.shape), 1)]).reshape((A, B, C))
        exit_wave[:] = exit_wave_fill

        mask_fill = np.ones_like(mask)
        # mask_fill[::2, ::2] = 0  # checkerboard for testing
        mask[:] = mask_fill

        pa = np.zeros((A, 3), dtype=np.int32)
        pa[:N,0] = 0
        pa[N:, 0] = 1

        X, Y = np.meshgrid(range(npts_greater_than),
                           range(npts_greater_than))  # assume square scan grid. Again, not always true.
        oa = np.zeros((A, 3), dtype=np.int32)
        oa[:N, 1] = X.ravel()
        oa[N:, 1] = X.ravel()
        oa[:N, 2] = Y.ravel()
        oa[N:, 2] = Y.ravel()


        ea = np.array([np.array([ix, 0, 0]) for ix in range(A)])
        da = np.array([np.array([ix, 0, 0]) for ix in range(N)] * D * G)
        ma = np.zeros((A, 3), dtype=np.int32)

        addr_info[:, 0, :] = pa
        addr_info[:, 1, :] = oa
        addr_info[:, 2, :] = ea
        addr_info[:, 3, :] = da
        addr_info[:, 4, :] = ma

        obj_weights = np.empty(shape=(G,), dtype=FLOAT_TYPE)
        obj_weights[:] = np.linspace(-1, 1, G)

        probe_weights = np.empty(shape=(D,), dtype=FLOAT_TYPE)
        probe_weights[:] = np.linspace(-1, 1, D)

        cfact_object = np.empty_like(obj)
        for idx in range(G):
            cfact_object[idx] = np.ones((H, I)) * 10 * (idx + 1)

        cfact_probe = np.empty_like(probe)

        for idx in range(D):
            cfact_probe[idx] = np.ones((B, C)) * 5 * (idx + 1)

        gexit_wave = deepcopy(exit_wave)
        gprobe = deepcopy(probe)
        gobj = deepcopy(obj)

        errors = con.difference_map_iterator(diffraction=diffraction,
                                             obj=obj,
                                             object_weights=obj_weights,
                                             cfact_object=cfact_object,
                                             mask=mask,
                                             probe=probe,
                                             cfact_probe=cfact_probe,
                                             probe_support=None,
                                             probe_weights=probe_weights,
                                             exit_wave=exit_wave,
                                             addr=addr_info,
                                             pre_fft=prefilter,
                                             post_fft=postfilter,
                                             pbound=pbound,
                                             overlap_max_iterations=10,
                                             update_object_first=True,
                                             obj_smooth_std=None,
                                             overlap_converge_factor=1.4e-3,
                                             probe_center_tol=None,
                                             probe_update_start=1,
                                             alpha=alpha,
                                             clip_object=None,
                                             LL_error=True,
                                             num_iterations=num_iter)

        gerrors = gcon.difference_map_iterator(diffraction=diffraction,
                                             obj=gobj,
                                             object_weights=obj_weights,
                                             cfact_object=cfact_object,
                                             mask=mask,
                                             probe=gprobe,
                                             cfact_probe=cfact_probe,
                                             probe_support=None,
                                             probe_weights=probe_weights,
                                             exit_wave=gexit_wave,
                                             addr=addr_info,
                                             pre_fft=prefilter,
                                             post_fft=postfilter,
                                             pbound=pbound,
                                             overlap_max_iterations=10,
                                             update_object_first=True,
                                             obj_smooth_std=None,
                                             overlap_converge_factor=1.4e-3,
                                             probe_center_tol=None,
                                             probe_update_start=1,
                                             alpha=alpha,
                                             clip_object=None,
                                             LL_error=True,
                                             num_iterations=num_iter)


        np.testing.assert_allclose(gerrors,
                                   errors,
                                   rtol=1e-6,
                                   err_msg="The returned errors are not consistent.")

        np.testing.assert_allclose(gprobe,
                                   probe,
                                   rtol=1e-6,
                                   err_msg="The returned probes are not consistent.")

        np.testing.assert_allclose(gobj,
                                   obj,
                                   rtol=1e-6,
                                   err_msg="The returned objects are not consistent.")

        np.testing.assert_allclose(gexit_wave,
                                   exit_wave,
                                   rtol=1e-6,
                                   err_msg="The returned exit_waves are not consistent.")


if __name__ == '__main__':
    unittest.main()



