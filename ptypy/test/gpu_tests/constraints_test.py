'''
The tests for the constraints
'''


import unittest
import utils as tu
import numpy as np
from ptypy.array_based import data_utils as du
from ptypy.array_based.constraints import difference_map_fourier_constraint, renormalise_fourier_magnitudes, get_difference
from ptypy.array_based.error_metrics import far_field_error
from ptypy.array_based.object_probe_interaction import difference_map_realspace_constraint, scan_and_multiply
from ptypy.array_based.propagation import farfield_propagator
import ptypy.array_based.array_utils as au
from ptypy.array_based import COMPLEX_TYPE, FLOAT_TYPE

from ptypy.gpu.constraints import get_difference as gget_difference
from ptypy.gpu.constraints import renormalise_fourier_magnitudes as grenormalise_fourier_magnitudes
from ptypy.gpu.constraints import difference_map_fourier_constraint as gdifference_map_fourier_constraint
from ptypy.gpu import array_utils as gau

class ConstraintsTest(unittest.TestCase):

    def test_get_difference_UNITY(self):
        alpha = 1.0
        pbound = None
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        master_pod = PtychoInstance.diff.V[first_view_id].pod
        propagator = master_pod.geometry.propagator
        addr = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        Idata = vectorised_scan['diffraction']
        mask = vectorised_scan['mask']

        view_dlayer = 0  # what is this?
        addr_info = addr[:, (view_dlayer)]  # addresses, object references
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
        addr = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        Idata = vectorised_scan['diffraction']
        mask = vectorised_scan['mask']

        view_dlayer = 0  # what is this?
        addr_info = addr[:, (view_dlayer)]  # addresses, object references
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
        addr = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        Idata = vectorised_scan['diffraction']
        mask = vectorised_scan['mask']

        view_dlayer = 0  # what is this?
        addr_info = addr[:, (view_dlayer)]  # addresses, object references
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
        addr = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        Idata = vectorised_scan['diffraction']
        mask = vectorised_scan['mask']

        view_dlayer = 0  # what is this?
        addr_info = addr[:, (view_dlayer)]  # addresses, object references
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
        addr = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        Idata = vectorised_scan['diffraction']
        mask = vectorised_scan['mask']

        view_dlayer = 0  # what is this?
        addr_info = addr[:, (view_dlayer)]  # addresses, object references
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
        addr = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        Idata = vectorised_scan['diffraction']
        mask = vectorised_scan['mask']

        view_dlayer = 0  # what is this?
        addr_info = addr[:, (view_dlayer)]  # addresses, object references
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

        np.testing.assert_allclose(exit_wave, gexit_wave, rtol=1e-1, atol=18)
        np.testing.assert_allclose(errors, gerrors, rtol=2e-4)

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

        np.testing.assert_allclose(exit_wave, gexit_wave, rtol=1e-1, atol=16)
        np.testing.assert_allclose(errors, gerrors, rtol=2e-4)


    def test_difference_map_fourier_constraint_no_update_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        master_pod = PtychoInstance.diff.V[first_view_id].pod
        propagator = master_pod.geometry.propagator
        pbound = 200.0

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

        np.testing.assert_allclose(exit_wave, gexit_wave, rtol=1e-1, atol=8)
        np.testing.assert_allclose(errors, gerrors, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()



