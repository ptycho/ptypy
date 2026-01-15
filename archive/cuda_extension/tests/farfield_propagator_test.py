'''
Test for the propagation in numpy
'''

import unittest
import numpy as np
from . import utils as tu
from ptypy.accelerate.array_based import data_utils as du
from ptypy.accelerate.array_based import object_probe_interaction as opi
from archive.array_based import propagation as prop
import time

from . import have_cuda, only_if_cuda_available
if have_cuda():
    from archive.cuda_extension.accelerate.cuda import propagation as gprop
    from archive.cuda_extension.accelerate.cuda.config import init_gpus, reset_function_cache
    init_gpus(0)

doTiming = False

def calculatePrintErrors(expected, actual):
    abserr = np.abs(expected-actual)
    max_abserr = np.max(abserr)
    mean_abserr = np.mean(abserr)
    min_abserr = np.min(abserr)
    std_abserr = np.std(abserr)
    relerr = abserr / np.abs(expected)
    max_relerr = np.nanmax(relerr)
    mean_relerr = np.nanmean(relerr)
    min_relerr = np.nanmin(relerr)
    std_relerr = np.nanstd(relerr)
    print("Abs Errors: max={}, min={}, mean={}, stddev={}".format(
        max_abserr, min_abserr, mean_abserr, std_abserr))
    print("Rel Errors: max={}, min={}, mean={}, stddev={}".format(
        max_relerr, min_relerr, mean_relerr, std_relerr))
    
@only_if_cuda_available
class FarfieldPropagatorTest(unittest.TestCase):

    def tearDown(self):
        # reset the cached GPU functions after each test
        reset_function_cache()

    #@unittest.skip("This method is not implemented yet")
    def test_fourier_transform_farfield_nofilter_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        exit_wave = opi.scan_and_multiply(vectorised_scan['probe'],
                                        vectorised_scan['obj'],
                                        vectorised_scan['exit wave'].shape,
                                        vectorised_scan['meta']['addr'])

        if doTiming: tstart = time.time()
        array_propagated = prop.farfield_propagator(exit_wave, prefilter=None, postfilter=None)
        if doTiming: 
            tend = time.time()
            pytime = tend-tstart
            tstart = time.time()
        gpu_propagated = gprop.farfield_propagator(exit_wave, prefilter=None, postfilter=None)
        if doTiming: 
            tend = time.time()
            gtime = tend-tstart

            print("Times: CPU={}, GPU={}, speedup={}x".format(
                pytime, gtime, pytime/gtime
            ))
        

        calculatePrintErrors(array_propagated, gpu_propagated)
        np.testing.assert_allclose(
            gpu_propagated, 
            array_propagated, rtol=1e-6, atol=3e-4,verbose=True
            )


    #@unittest.skip("This method is not implemented yet")
    def test_fourier_transform_farfield_with_prefilter_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        propagator = PtychoInstance.di.V[first_view_id].pod.geometry.propagator
        exit_wave = opi.scan_and_multiply(vectorised_scan['probe'],
                                        vectorised_scan['obj'],
                                        vectorised_scan['exit wave'].shape,
                                        vectorised_scan['meta']['addr'])

        array_propagated = prop.farfield_propagator(exit_wave, prefilter=propagator.pre_fft, postfilter=None)
        gpu_propagated = gprop.farfield_propagator(exit_wave, prefilter=propagator.pre_fft, postfilter=None)

        calculatePrintErrors(array_propagated, gpu_propagated)
        np.testing.assert_allclose(
            gpu_propagated, 
            array_propagated, rtol=1e-6, atol=4e-4,verbose=True
            )

    #@unittest.skip("This method is not implemented yet")
    def test_fourier_transform_farfield_with_postfilter_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        propagator = PtychoInstance.di.V[first_view_id].pod.geometry.propagator
        exit_wave = opi.scan_and_multiply(vectorised_scan['probe'],
                                        vectorised_scan['obj'],
                                        vectorised_scan['exit wave'].shape,
                                        vectorised_scan['meta']['addr'])

        array_propagated = prop.farfield_propagator(exit_wave, prefilter=None, postfilter=propagator.post_fft)
        gpu_propagated = gprop.farfield_propagator(exit_wave, prefilter=None, postfilter=propagator.post_fft)

        calculatePrintErrors(array_propagated, gpu_propagated)
        np.testing.assert_allclose(
            gpu_propagated, 
            array_propagated, rtol=1e-6, atol=3e-4,verbose=True
            )

    #@unittest.skip("This method is not implemented yet")
    def test_fourier_transform_farfield_with_pre_and_post_filter_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        propagator = PtychoInstance.di.V[first_view_id].pod.geometry.propagator
        exit_wave = opi.scan_and_multiply(vectorised_scan['probe'],
                                        vectorised_scan['obj'],
                                        vectorised_scan['exit wave'].shape,
                                        vectorised_scan['meta']['addr'])
        if doTiming: tstart = time.time()
        array_propagated = prop.farfield_propagator(exit_wave, prefilter=propagator.pre_fft, postfilter=propagator.post_fft)
        if doTiming: 
            tend = time.time()
            pytime = tend-tstart
            tstart = time.time()
        gpu_propagated = gprop.farfield_propagator(exit_wave, prefilter=propagator.pre_fft, postfilter=propagator.post_fft)
        if doTiming: 
            tend = time.time()
            gtime = tend-tstart

            print("Times: CPU={}, GPU={}, speedup={}x".format(
                pytime, gtime, pytime/gtime
            ))
        

        calculatePrintErrors(array_propagated, gpu_propagated)
        np.testing.assert_allclose(
            gpu_propagated, 
            array_propagated, rtol=1e-6, atol=5e-4,verbose=True
            )

    #@unittest.skip("This method is not implemented yet")
    def test_inverse_fourier_transform_farfield_nofilter_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000',)
        exit_wave = opi.scan_and_multiply(vectorised_scan['probe'],
                                        vectorised_scan['obj'],
                                        vectorised_scan['exit wave'].shape,
                                        vectorised_scan['meta']['addr'])

        array_propagated = prop.farfield_propagator(exit_wave, prefilter=None, postfilter=None, direction='backward')
        gpu_propagated = gprop.farfield_propagator(exit_wave, prefilter=None, postfilter=None, direction='backward')

        calculatePrintErrors(array_propagated, gpu_propagated)
        np.testing.assert_allclose(
            gpu_propagated, 
            array_propagated, rtol=1e-6, atol=5e-4,verbose=True
            )

    #@unittest.skip("This method is not implemented yet")
    def test_inverse_fourier_transform_farfield_with_prefilter_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        propagator = PtychoInstance.di.V[first_view_id].pod.geometry.propagator
        exit_wave = opi.scan_and_multiply(vectorised_scan['probe'],
                                        vectorised_scan['obj'],
                                        vectorised_scan['exit wave'].shape,
                                        vectorised_scan['meta']['addr'])

        array_propagated = prop.farfield_propagator(exit_wave, prefilter=propagator.pre_fft, postfilter=None, direction='backward')
        gpu_propagated = gprop.farfield_propagator(exit_wave, prefilter=propagator.pre_fft, postfilter=None, direction='backward')

        calculatePrintErrors(array_propagated, gpu_propagated)
        np.testing.assert_allclose(
            gpu_propagated, 
            array_propagated, rtol=1e-6, atol=5e-4,verbose=True
            )

    #@unittest.skip("This method is not implemented yet")
    def test_inverse_fourier_transform_farfield_with_postfilter_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        propagator = PtychoInstance.di.V[first_view_id].pod.geometry.propagator
        exit_wave = opi.scan_and_multiply(vectorised_scan['probe'],
                                        vectorised_scan['obj'],
                                        vectorised_scan['exit wave'].shape,
                                        vectorised_scan['meta']['addr'])

        array_propagated = prop.farfield_propagator(exit_wave, prefilter=None, postfilter=propagator.post_fft, direction='backward')
        gpu_propagated = gprop.farfield_propagator(exit_wave, prefilter=None, postfilter=propagator.post_fft, direction='backward')

        calculatePrintErrors(array_propagated, gpu_propagated)
        np.testing.assert_allclose(
            gpu_propagated, 
            array_propagated, rtol=1e-6, atol=5e-4,verbose=True
            )

    #@unittest.skip("This method is not implemented yet")
    def test_inverse_fourier_transform_farfield_with_pre_and_post_filter_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        propagator = PtychoInstance.di.V[first_view_id].pod.geometry.propagator
        exit_wave = opi.scan_and_multiply(vectorised_scan['probe'],
                                        vectorised_scan['obj'],
                                        vectorised_scan['exit wave'].shape,
                                        vectorised_scan['meta']['addr'])

        array_propagated = prop.farfield_propagator(exit_wave, prefilter=propagator.pre_fft, postfilter=propagator.post_fft, direction='backward')
        gpu_propagated = gprop.farfield_propagator(exit_wave, prefilter=propagator.pre_fft, postfilter=propagator.post_fft, direction='backward')

        calculatePrintErrors(array_propagated, gpu_propagated)
        np.testing.assert_allclose(
            gpu_propagated, 
            array_propagated, rtol=1e-6, atol=5e-4,verbose=True
            )

#

if __name__ == "__main__":
    unittest.main()
