'''
The tests for the constraints
'''


import unittest
import numpy as np
import utils as tu
from ptypy.gpu import data_utils as du
from collections import OrderedDict
from ptypy.engines.utils import basic_fourier_update
import ptypy.engines.utils as u
from ptypy.gpu.constraints import difference_map_fourier_constraint, renormalise_fourier_magnitudes, get_difference
from ptypy.gpu.error_metrics import far_field_error
from ptypy.gpu.object_probe_interaction import difference_map_realspace_constraint, scan_and_multiply
from ptypy.gpu.propagation import farfield_propagator
import ptypy.gpu.array_utils as au
import ptypy.utils as u
from ptypy.gpu import COMPLEX_TYPE, FLOAT_TYPE

class ConstraintsTest(unittest.TestCase):

    def test_get_difference(self):
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
        constrained = difference_map_realspace_constraint(obj, probe, exit_wave, addr, alpha)
        f = farfield_propagator(constrained, propagator.pre_fft, propagator.post_fft, direction='forward')
        pa, oa, ea, da, ma = zip(*addr_info)
        af2 = au.sum_to_buffer(au.abs2(f), Idata.shape, ea, da, dtype=FLOAT_TYPE)

        fmag = np.sqrt(np.abs(Idata))
        af = np.sqrt(af2)
        # # Fourier magnitudes deviations(current_solution, pbound, measured_solution, mask, addr)
        err_fmag = far_field_error(af, fmag, mask)
        renormed_f = renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound)
        probe_object = scan_and_multiply(probe, obj, exit_wave.shape, addr_info)

        backpropagated_solution = farfield_propagator(renormed_f,
                                                      propagator.post_fft.conj(),
                                                      propagator.pre_fft.conj(),
                                                      direction='backward')

        get_difference(addr_info, alpha, backpropagated_solution, err_fmag, exit_wave, pbound, probe_object)

    def test_get_difference_UNITY(self):
        alpha = 1.0
        pbound = None

        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        master_pod = PtychoInstance.diff.V[first_view_id].pod
        propagator = master_pod.geometry.propagator

        addr = vectorised_scan['meta']['addr']
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        Idata = vectorised_scan['diffraction']
        mask = vectorised_scan['mask']

        view_dlayer = 0  # what is this?
        addr_info = addr[:, (view_dlayer)]  # addresses, object references
        # # Propagate the exit waves
        constrained = difference_map_realspace_constraint(obj, probe, exit_wave, addr, alpha)
        f = farfield_propagator(constrained, propagator.pre_fft, propagator.post_fft, direction='forward')
        pa, oa, ea, da, ma = zip(*addr_info)
        af2 = au.sum_to_buffer(au.abs2(f), Idata.shape, ea, da, dtype=FLOAT_TYPE)

        fmag = np.sqrt(np.abs(Idata))
        af = np.sqrt(af2)
        # # Fourier magnitudes deviations(current_solution, pbound, measured_solution, mask, addr)
        err_fmag = far_field_error(af, fmag, mask)
        err_fmag = np.ones_like(err_fmag) * 145.824958919
        vectorised_rfm = renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound)

        probe_object = scan_and_multiply(probe, obj, exit_wave.shape, addr_info)

        backpropagated_solution = farfield_propagator(vectorised_rfm,
                                                      propagator.pre_ifft,
                                                      propagator.post_ifft,
                                                      direction='backward')

        df = get_difference(addr_info, alpha, backpropagated_solution, err_fmag, exit_wave, pbound, probe_object)
        POD_ptycho_instance = tu.get_ptycho_instance('pod_to_numpy_test')
        ptypy_result = self.ptypy_get_renormalised_fourier_magnitudes(POD_ptycho_instance, pbound=pbound)

        np.testing.assert_array_equal(ptypy_result['differences'],
                                       df,
                                       err_msg='The POD and vectorised version of the difference map are not consistent')

    def test_get_difference_Pbound_UNITY(self):
        alpha = 1.0
        pbound = 0.597053604126

        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        master_pod = PtychoInstance.diff.V[first_view_id].pod
        propagator = master_pod.geometry.propagator

        addr = vectorised_scan['meta']['addr']
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        Idata = vectorised_scan['diffraction']
        mask = vectorised_scan['mask']

        view_dlayer = 0  # what is this?
        addr_info = addr[:, (view_dlayer)]  # addresses, object references
        # # Propagate the exit waves
        constrained = difference_map_realspace_constraint(obj, probe, exit_wave, addr, alpha)
        f = farfield_propagator(constrained, propagator.pre_fft, propagator.post_fft, direction='forward')
        pa, oa, ea, da, ma = zip(*addr_info)
        af2 = au.sum_to_buffer(au.abs2(f), Idata.shape, ea, da, dtype=FLOAT_TYPE)

        fmag = np.sqrt(np.abs(Idata))
        af = np.sqrt(af2)
        # # Fourier magnitudes deviations(current_solution, pbound, measured_solution, mask, addr)
        err_fmag = far_field_error(af, fmag, mask)
        err_fmag = np.ones_like(err_fmag) * 145.824958919
        vectorised_rfm = renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound)

        probe_object = scan_and_multiply(probe, obj, exit_wave.shape, addr_info)

        backpropagated_solution = farfield_propagator(vectorised_rfm,
                                                      propagator.pre_ifft,
                                                      propagator.post_ifft,
                                                      direction='backward')

        df = get_difference(addr_info, alpha, backpropagated_solution, err_fmag, exit_wave, pbound, probe_object)
        POD_ptycho_instance = tu.get_ptycho_instance('pod_to_numpy_test')
        ptypy_result = self.ptypy_get_renormalised_fourier_magnitudes(POD_ptycho_instance, pbound=pbound, ferr=145.824958919)

        np.testing.assert_array_equal(vectorised_rfm,
                                      ptypy_result['renormalised'])


        np.testing.assert_array_equal(ptypy_result['differences'],
                                       df,
                                       err_msg='The POD and vectorised version of the difference map are not consistent')

    def test_renormalise_fourier_magnitudes(self):
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
        constrained = difference_map_realspace_constraint(obj, probe, exit_wave, addr, alpha)
        f = farfield_propagator(constrained, propagator.pre_fft, propagator.post_fft, direction='forward')
        pa, oa, ea, da, ma = zip(*addr_info)
        af2 = au.sum_to_buffer(au.abs2(f), Idata.shape, ea, da, dtype=FLOAT_TYPE)

        fmag = np.sqrt(np.abs(Idata))
        af = np.sqrt(af2)
        # # Fourier magnitudes deviations(current_solution, pbound, measured_solution, mask, addr)
        err_fmag = far_field_error(af, fmag, mask)
        renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound)

    def test_renormalise_fourier_magnitudes_UNITY(self):
        alpha = 1.0
        pbound = None

        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        master_pod = PtychoInstance.diff.V[first_view_id].pod
        propagator = master_pod.geometry.propagator

        addr = vectorised_scan['meta']['addr']
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        Idata = vectorised_scan['diffraction']
        mask = vectorised_scan['mask']

        view_dlayer = 0  # what is this?
        addr_info = addr[:, (view_dlayer)]  # addresses, object references
        # # Propagate the exit waves
        constrained = difference_map_realspace_constraint(obj, probe, exit_wave, addr, alpha)
        f = farfield_propagator(constrained, propagator.pre_fft, propagator.post_fft, direction='forward')
        pa, oa, ea, da, ma = zip(*addr_info)
        af2 = au.sum_to_buffer(au.abs2(f), Idata.shape, ea, da, dtype=FLOAT_TYPE)

        fmag = np.sqrt(np.abs(Idata))
        af = np.sqrt(af2)
        # # Fourier magnitudes deviations(current_solution, pbound, measured_solution, mask, addr)
        err_fmag = far_field_error(af, fmag, mask)
        err_fmag = np.ones_like(err_fmag)* 145.824958919
        vectorised_rfm = renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound)
        ptypy_result = self.ptypy_get_renormalised_fourier_magnitudes(tu.get_ptycho_instance('pod_to_numpy_test'),
                                                                      pbound,
                                                                      ferr=145.824958919)


        # double check the inputs
        print ptypy_result['renormalised'].dtype, vectorised_rfm.dtype
        np.testing.assert_array_equal(ptypy_result['renormalised'],
                                    vectorised_rfm,
                                    err_msg="The POD based and vectorised fourier magnitude renomalisation is not consistent.")

    def test_renormalise_fourier_magnitudes_pbound_UNITY(self):
        alpha = 1.0
        pbound = 0.597053604126

        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        master_pod = PtychoInstance.diff.V[first_view_id].pod
        propagator = master_pod.geometry.propagator

        addr = vectorised_scan['meta']['addr']
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        Idata = vectorised_scan['diffraction']
        mask = vectorised_scan['mask']

        view_dlayer = 0  # what is this?
        addr_info = addr[:, (view_dlayer)]  # addresses, object references
        # # Propagate the exit waves
        constrained = difference_map_realspace_constraint(obj, probe, exit_wave, addr, alpha)
        f = farfield_propagator(constrained, propagator.pre_fft, propagator.post_fft, direction='forward')
        pa, oa, ea, da, ma = zip(*addr_info)
        af2 = au.sum_to_buffer(au.abs2(f), Idata.shape, ea, da, dtype=FLOAT_TYPE)

        fmag = np.sqrt(np.abs(Idata))
        af = np.sqrt(af2)
        # # Fourier magnitudes deviations(current_solution, pbound, measured_solution, mask, addr)
        err_fmag = far_field_error(af, fmag, mask)
        err_fmag = np.ones_like(err_fmag)* 145.824958919
        vectorised_rfm = renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound)
        ptypy_result = self.ptypy_get_renormalised_fourier_magnitudes(tu.get_ptycho_instance('pod_to_numpy_test'),
                                                                      pbound,
                                                                      ferr=145.824958919)

        # double check the inputs

        np.testing.assert_array_equal(ptypy_result['renormalised'],
                                    vectorised_rfm,
                                    err_msg="The POD based and vectorised fourier magnitude renomalisation is not consistent.")

    def test_difference_map_fourier_constraint(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        # now convert to arrays
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        addr = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        diffraction = vectorised_scan['diffraction']
        mask = vectorised_scan['mask']
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        master_pod = PtychoInstance.diff.V[first_view_id].pod
        propagator = master_pod.geometry.propagator
        difference_map_fourier_constraint(mask,
                                          diffraction,
                                          obj,
                                          probe,
                                          exit_wave,
                                          addr,
                                          prefilter=propagator.pre_fft,
                                          postfilter=propagator.post_fft,
                                          pbound=None,
                                          alpha=1.0,
                                          LL_error=True)

    def test_difference_map_fourier_constraint_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        PodPtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')

        # now convert to arrays
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')

        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        master_pod = PtychoInstance.diff.V[first_view_id].pod
        propagator = master_pod.geometry.propagator

        ptypy_ewf, ptypy_error= self.ptypy_difference_map_fourier_constraint(PodPtychoInstance)
        exit_wave, errors = difference_map_fourier_constraint(vectorised_scan['mask'],
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

        for idx, key in enumerate(ptypy_ewf.keys()):
            np.testing.assert_array_equal(ptypy_ewf[key],
                                       exit_wave[idx],
                                       err_msg="The array-based and pod-based exit waves are not consistent")

        ptypy_fmag = []
        ptypy_phot = []
        ptypy_exit = []

        for idx, key in enumerate(ptypy_error.keys()):
            err_fmag, err_phot, err_exit = ptypy_error[key]
            ptypy_fmag.append(err_fmag)
            ptypy_phot.append(err_phot)
            ptypy_exit.append(err_exit)

        ptypy_fmag = np.array(ptypy_fmag)
        ptypy_phot = np.array(ptypy_phot)
        ptypy_exit = np.array(ptypy_exit)

        npy_fmag = errors[0, :]
        npy_phot = errors[1, :]
        npy_exit = errors[2, :]

        np.testing.assert_array_equal(npy_fmag,
                                   ptypy_fmag,
                                   err_msg="The array-based and pod-based fmag errors are not consistent")

        np.testing.assert_array_equal(npy_phot,
                                   ptypy_phot,
                                   err_msg="The array-based and pod-based phot errors are not consistent")
        # there is a slight difference in numpy in the way the mean is calculated here.  It 1e-13 and a diagnostic so almost equal is fine
        np.testing.assert_array_almost_equal(npy_exit,
                                   ptypy_exit,
                                   err_msg="The array-based and pod-based exit errors are not consistent")

    def ptypy_difference_map_fourier_constraint(self, a_ptycho_instance):
        error_dct = OrderedDict()
        exit_wave = OrderedDict()
        for dname, diff_view in a_ptycho_instance.diff.views.iteritems():
            di_view = a_ptycho_instance.diff.V[dname]
            error_dct[dname] = basic_fourier_update(di_view,
                                                    pbound=None,
                                                    alpha=1.0)
            for name, pod in di_view.pods.iteritems():
                exit_wave[name] = pod.exit
        return exit_wave, error_dct

    def ptypy_get_renormalised_fourier_magnitudes(self, a_ptycho_instance, pbound, ferr=None):
        out_dict = {}
        alpha = 1.0
        renormalised = []
        differences = []
        backpropagated = []
        probe_object = []
        fm_out = []
        fmag_out = []
        af_out = []
        for name, diff_view in a_ptycho_instance.di.views.iteritems():
            f = {}

            # Buffer for accumulated photons
            af2 = np.zeros_like(diff_view.data)

            # Get measured data
            I = diff_view.data

            # Get the mask
            fmask = diff_view.pod.mask
            # Propagate the exit waves
            for name, pod in diff_view.pods.iteritems():
                f[name] = pod.fw((1 + alpha) * pod.probe * pod.object
                                 - alpha * pod.exit)
                af2 += u.abs2(f[name])

            fmag = np.sqrt(np.abs(I))
            fmag_out.append(fmag)
            af = np.sqrt(af2)
            af_out.append(af)
            # Fourier magnitudes deviations
            fdev = af - fmag
            if ferr is None:
                err_fmag = np.sum(fmask * fdev ** 2) / fmask.sum()
            else:
                err_fmag = ferr

            if pbound is None:
                # No power bound
                fm = (1 - fmask) + fmask * fmag / (af + 1e-10)
                for name, pod in diff_view.pods.iteritems():
                    renormalised_fourier_space = (fm * f[name])
                    backpropagated_solution = pod.bw(renormalised_fourier_space)
                    backpropagated.append(backpropagated_solution)
                    probe_object_mult = pod.probe * pod.object
                    probe_object.append(probe_object_mult)
                    df = backpropagated_solution - probe_object_mult
                    renormalised.append(renormalised_fourier_space)
                    differences.append(df)

            elif err_fmag > pbound:
                # Power bound is applied
                renorm = np.sqrt(pbound / err_fmag)

                fm =  (1 - fmask) + fmask * (fmag + fdev * renorm)/ (af + 1e-10)

                for name, pod in diff_view.pods.iteritems():
                    renormalised_fourier_space = (fm * f[name])
                    backpropagated_solution = pod.bw(renormalised_fourier_space)
                    backpropagated.append(backpropagated_solution)
                    probe_object_mult = pod.probe * pod.object
                    df = backpropagated_solution - probe_object_mult
                    probe_object.append(probe_object_mult)
                    renormalised.append(renormalised_fourier_space)
                    differences.append(df)

            else:
                # Within power bound so no constraint applied.
                for name, pod in diff_view.pods.iteritems():
                    backpropagated_solution = np.zeros(fm.shape, dtype=COMPLEX_TYPE)
                    backpropagated.append(backpropagated_solution)
                    renormalised_fourier_space = np.zeros(f[name].shape, dtype=COMPLEX_TYPE)
                    probe_object_mult = pod.probe * pod.object
                    df = alpha * (probe_object_mult - pod.exit)
                    probe_object.append(probe_object_mult)
                    renormalised.append(renormalised_fourier_space)
                    differences.append(df)
            fm_out.append(fm)

        out_dict['fmag'] = np.array(fmag_out)
        out_dict['fm'] = np.array(fm_out)
        out_dict['af'] = np.array(af_out)
        out_dict['renormalised'] = np.array(renormalised)
        out_dict['backpropagated'] = np.array(backpropagated)
        out_dict['differences'] = np.array(differences)
        out_dict['probe_object'] = np.array(probe_object)
        return  out_dict

if __name__ == '__main__':
    unittest.main()



