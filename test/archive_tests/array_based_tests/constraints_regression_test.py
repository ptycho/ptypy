'''
The tests for the constraints
'''


import unittest
import numpy as np
from copy import deepcopy
from ptypy.accelerate.array_based import constraints as con, FLOAT_TYPE, COMPLEX_TYPE

class ConstraintsRegressionTest(unittest.TestCase):
    '''
    a module to holds the constraints
    '''

    def test_renormalise_fourier_magnitudes_pbound_none(self):
        num_object_modes = 1 # for example
        num_probe_modes = 2 # for example

        N = 3 # number of measured points
        M = N * num_object_modes * num_probe_modes # exit wave length
        A = 2 # for example
        B = 4 # for example
        pbound = None  # the power bound

        fmag = np.empty(shape=(N, A, B), dtype=FLOAT_TYPE)# the measured magnitudes NxAxB
        mask = np.empty(shape=(N, A, B), dtype=np.int32)# the masks for the measured magnitudes either 1xAxB or NxAxB
        err_fmag = np.empty(shape=(N, ), dtype=FLOAT_TYPE)# deviation from the diffraction pattern for each af
        f = np.empty(shape=(M, A, B), dtype=COMPLEX_TYPE) # the current iterant
        af = np.empty(shape=(M, A, B), dtype=FLOAT_TYPE)# the absolute magnitudes of f
        addr_info = np.empty(shape=(M, 5, 3), dtype=np.int32)# the address book

        ## now lets fill them with some values, these are junk and not supposed to be indicative of real values.

        fmag_fill = np.arange(np.prod(fmag.shape)).reshape(fmag.shape).astype(fmag.dtype)
        fmag[:] = fmag_fill

        mask_fill = np.ones_like(mask)
        mask_fill[::2, ::2] = 0 # checkerboard for testing
        mask[:] = mask_fill

        err_fmag[:] = np.ones((N,)) # this shouldn't be used a pbound is None

        f_fill = np.array([ix + 1j*(ix**2) for ix in range(np.prod(f.shape))]).reshape((M, A, B))
        f[:] = f_fill

        af[:] = np.sqrt((f*f.conj()).real)
        pa = np.zeros((M, 3), dtype=np.int32) # not going to be used here
        oa = np.zeros((M, 3), dtype=np.int32) # not going to be used here
        ea = np.array([np.array([ix, 0, 0]) for ix in range(M)])
        da = np.array([np.array([ix, 0, 0]) for ix in range(N)]*num_probe_modes*num_object_modes)
        ma = np.array([np.array([ix, 0, 0]) for ix in range(N)]*num_probe_modes*num_object_modes)

        addr_info[:, 0, :] = pa
        addr_info[:, 1, :] = oa
        addr_info[:, 2, :] = ea
        addr_info[:, 3, :] = da
        addr_info[:, 4, :] = ma
        expected_out = np.array([[[0.0 + 0.0j, 1.0 + 1.0j, 2.0 + 4.0j, 3.0 + 9.0j],
                          [0.97014252 + 3.88057009j, 0.98058065 + 4.90290327e+00j, 0.98639394 + 5.91836367e+00j, 0.98994949 + 6.92964646e+00j]],
                         [[0.99227787 + 7.93822300j, 0.99388373 + 8.94495358j, 0.99503719 + 9.95037188j, 0.99589322 + 10.9548254j],
                          [0.99654579 + 1.19585495e+01j, 0.99705446 + 1.29617079e+01j, 0.99745872 + 1.39644221e+01j, 0.99778514 + 1.49667770e+01j]],
                         [[16.00000000 + 2.56000000e+02j, 17.00000000 + 2.89000000e+02j, 18.00000000 + 3.24000000e+02j,19.00000000 + 3.61000000e+02j],
                          [0.99875232 + 1.99750464e+01j, 0.99886812 + 2.09762305e+01j, 0.99896851 + 2.19773073e+01j, 0.99905617 + 2.29782920e+01j]],
                         [[24.00000000 + 5.76000000e+02j, 25.00000000 + 6.25000000e+02j, 26.00000000 + 6.76000000e+02j, 27.00000000 + 7.29000000e+02j],
                          [6.79099767 + 1.90147935e+02j, 5.68736780 + 1.64933666e+02j, 4.93196972 + 1.47959092e+02j, 4.38406204 + 1.35905923e+02j]],
                         [[3.96911150 + 1.27011568e+02j, 3.64424035 + 1.20259932e+02j, 3.38312644 + 1.15026299e+02j, 3.16875114 + 1.10906290e+02j],
                          [2.98963737 + 1.07626945e+02j, 2.83777037 + 1.04997504e+02j, 2.70738796 + 1.02880743e+02j, 2.59424135 + 1.01175413e+02j]],
                         [[40.00000000 + 1.60000000e+03j, 41.00000000 + 1.68100000e+03j, 42.00000000 + 1.76400000e+03j, 43.00000000 + 1.84900000e+03j],
                          [2.19725511 + 9.66792247e+01j, 2.14043168 + 9.63194257e+01j, 2.08875234 + 9.60826078e+01j, 2.04154957 + 9.59528299e+01j]]], dtype=COMPLEX_TYPE)


        out = con.renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound)
        np.testing.assert_allclose(out, expected_out, rtol=1e-6)


    def test_renormalise_fourier_magnitudes_pbound_not_none(self):
        num_object_modes = 1 # for example
        num_probe_modes = 2 # for example

        N = 3 # number of measured points
        M = N * num_object_modes * num_probe_modes # exit wave length
        A = 2 # for example
        B = 4 # for example
        pbound = 5.0  # the power bound

        fmag = np.empty(shape=(N, A, B), dtype=FLOAT_TYPE)# the measured magnitudes NxAxB
        mask = np.empty(shape=(N, A, B), dtype=np.int32)# the masks for the measured magnitudes either 1xAxB or NxAxB
        err_fmag = np.empty(shape=(N, ), dtype=FLOAT_TYPE)# deviation from the diffraction pattern for each af
        f = np.empty(shape=(M, A, B), dtype=COMPLEX_TYPE) # the current iterant
        af = np.empty(shape=(M, A, B), dtype=FLOAT_TYPE)# the absolute magnitudes of f
        addr_info = np.empty(shape=(M, 5, 3), dtype=np.int32)# the address book

        ## now lets fill them with some values, these are junk and not supposed to be indicative of real values.

        fmag_fill = np.arange(np.prod(fmag.shape)).reshape(fmag.shape).astype(fmag.dtype)
        fmag[:] = fmag_fill

        mask_fill = np.ones_like(mask)
        mask_fill[::2, ::2] = 0 # checkerboard for testing
        mask[:] = mask_fill

        err_fmag_fill = np.ones((N,))*(pbound+0.1) # should be greater than the pbound
        err_fmag_fill[N//2] = 4.0 #  this one should be less than the pbound and not update
        err_fmag[:] = err_fmag_fill  # this shouldn't be used a pbound is None

        f_fill = np.array([ix + 1j*(ix**2) for ix in range(np.prod(f.shape))]).reshape((M, A, B))
        f[:] = f_fill

        af[:] = np.sqrt((f*f.conj()).real)
        pa = np.zeros((M, 3), dtype=np.int32) # not going to be used here
        oa = np.zeros((M, 3), dtype=np.int32) # not going to be used here
        ea = np.array([np.array([ix, 0, 0]) for ix in range(M)])
        da = np.array([np.array([ix, 0, 0]) for ix in range(N)]*num_probe_modes*num_object_modes)
        ma = np.array([np.array([ix, 0, 0]) for ix in range(N)]*num_probe_modes*num_object_modes)

        addr_info[:, 0, :] = pa
        addr_info[:, 1, :] = oa
        addr_info[:, 2, :] = ea
        addr_info[:, 3, :] = da
        addr_info[:, 4, :] = ma
        expected_out = np.array([[[0.0 +0.0j, 1.0 +1.0j, 2.0 +4.0j, 3.0 +9.0j],
                                  [3.97014832 + 1.58805933e+01j, 4.96039867 +2.48019943e+01j, 5.95060349 +3.57036209e+01j, 6.94078636 +4.85855026e+01j]],
                                 [[0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                                  [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]],
                                 [[16.0 + 2.56e+02j, 17.0 + 2.89e+02j, 18.0 + 3.24e+02j, 19.0 + 3.61e+02j],
                                  [19.81278992 + 3.96255798e+02j, 20.80293846 +4.36861725e+02j, 21.79308891 + 4.79447937e+02j, 22.78323746 + 5.24014465e+02j]],
                                 [[ 24.0 + 5.76e+02j, 25.0 +6.25e+02j, 26.0 +6.76e+02j, 27.0 + 7.29e+02j],
                                  [27.79103851 +7.78149109e+02j, 28.77031326 +8.34339111e+02j, 29.75301552 + 8.92590515e+02j, 30.73776817 + 9.52870789e+02j]],
                                 [[0.0 + 0.0j, 0.00 + 0.00e+00j, 0.00 +0.00e+00j, 0.00 + 0.00e+00j],
                                  [0.00 +0.0j, 0.00 + 0.00j, 0.00 + 0.00j,   0.00 + 0.00j]],
                                 [[40.00 +1.60e+03j, 41.00 + 1.681e+03j, 42.0 +1.764e+03j, 43.0 + 1.849e+03j],
                                  [43.58813858 + 1.91787805e+03j, 44.57772446 + 2.00599768e+03j, 45.56736755 +2.09609888e+03j, 46.55705261 + 2.18818140e+03j]]], dtype=COMPLEX_TYPE)


        out = con.renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound)
        np.testing.assert_allclose(out, expected_out)

    def test_get_difference_pbound_is_none(self):
        alpha = 1.0 # feedback constant
        pbound = 5.0  # the power bound
        num_object_modes = 1 # for example
        num_probe_modes = 2 # for example

        N = 3 # number of measured points
        M = N * num_object_modes * num_probe_modes # exit wave length
        A = 2 # for example
        B = 4 # for example

        backpropagated_solution = np.empty(shape=(M, A, B), dtype=COMPLEX_TYPE)# The current iterant backpropagated
        probe_object = np.empty(shape=(M, A, B), dtype=COMPLEX_TYPE)# the probe multiplied by the object
        err_fmag = np.empty(shape=(N,), dtype=FLOAT_TYPE)# deviation from the diffraction pattern for each af
        exit_wave = np.empty(shape=(M, A, B), dtype=COMPLEX_TYPE) # exit wave
        addr_info = np.empty(shape=(M, 5, 3), dtype=np.int32)# the address book

        # now fill it with stuff

        backpropagated_solution_fill = np.array([ix + 1j*(ix**2) for ix in range(np.prod(backpropagated_solution.shape))]).reshape((M, A, B))
        backpropagated_solution[:] = backpropagated_solution_fill

        probe_object_fill = np.array([ix + 1j*ix for ix in range(10, 10+np.prod(backpropagated_solution.shape), 1)]).reshape((M, A, B))
        probe_object[:] = probe_object_fill

        err_fmag_fill = np.ones((N,))
        err_fmag[:] = err_fmag_fill  # this shouldn't be used as pbound is None

        exit_wave_fill =  np.array([ix**2 + 1j*ix for ix in range(20, 20+np.prod(backpropagated_solution.shape), 1)]).reshape((M, A, B))
        exit_wave[:] = exit_wave_fill

        pa = np.zeros((M, 3), dtype=np.int32) # not going to be used here
        oa = np.zeros((M, 3), dtype=np.int32) # not going to be used here
        ea = np.array([np.array([ix, 0, 0]) for ix in range(M)])
        da = np.array([np.array([ix, 0, 0]) for ix in range(N)]*num_probe_modes*num_object_modes)
        ma = np.zeros((M, 3), dtype=np.int32)

        addr_info[:, 0, :] = pa
        addr_info[:, 1, :] = oa
        addr_info[:, 2, :] = ea
        addr_info[:, 3, :] = da
        addr_info[:, 4, :] = ma

        expected_out = np.array([[[-390.-10.j, -430.-10.j, -472.-10.j, -516.-10.j],
                                  [-562.-10.j, -610.-10.j, -660.-10.j, -712.-10.j]],
                                 [[-766.-10.j, -822.-10.j, -880.-10.j, -940.-10.j],
                                  [-1002.-10.j, -1066.-10.j, -1132.-10.j, -1200.-10.j]],
                                 [[-1270.-10.j, -1342.-10.j, -1416.-10.j, -1492.-10.j],
                                  [-1570.-10.j, -1650.-10.j, -1732.-10.j, -1816.-10.j]],
                                 [[-1902.-10.j, -1990.-10.j, -2080.-10.j, -2172.-10.j],
                                  [-2266.-10.j, -2362.-10.j, -2460.-10.j, -2560.-10.j]],
                                 [[-2662.-10.j, -2766.-10.j, -2872.-10.j, -2980.-10.j],
                                  [-3090.-10.j, -3202.-10.j, -3316.-10.j, -3432.-10.j]],
                                 [[-3550.-10.j, -3670.-10.j, -3792.-10.j, -3916.-10.j],
                                  [-4042.-10.j, -4170.-10.j, -4300.-10.j, -4432.-10.j]]], dtype=COMPLEX_TYPE)

        out = con.get_difference(addr_info, alpha, backpropagated_solution, err_fmag, exit_wave, pbound, probe_object)
        np.testing.assert_allclose(expected_out, out)

    def test_get_difference_pbound_is_not_none(self):
        alpha = 1.0  # feedback constant
        pbound = 5.0  # the power bound
        num_object_modes = 1  # for example
        num_probe_modes = 2  # for example

        N = 3  # number of measured points
        M = N * num_object_modes * num_probe_modes  # exit wave length
        A = 2  # for example
        B = 4  # for example

        backpropagated_solution = np.empty(shape=(M, A, B),
                                           dtype=COMPLEX_TYPE)  # The current iterant backpropagated
        probe_object = np.empty(shape=(M, A, B), dtype=COMPLEX_TYPE)  # the probe multiplied by the object
        err_fmag = np.empty(shape=(N,), dtype=FLOAT_TYPE)  # deviation from the diffraction pattern for each af
        exit_wave = np.empty(shape=(M, A, B), dtype=COMPLEX_TYPE)  # exit wave
        addr_info = np.empty(shape=(M, 5, 3), dtype=np.int32)  # the address book

        # now fill it with stuff

        backpropagated_solution_fill = np.array(
            [ix + 1j * (ix ** 2) for ix in range(np.prod(backpropagated_solution.shape))]).reshape((M, A, B))
        backpropagated_solution[:] = backpropagated_solution_fill

        probe_object_fill = np.array(
            [ix + 1j * ix for ix in range(10, 10 + np.prod(backpropagated_solution.shape), 1)]).reshape((M, A, B))
        probe_object[:] = probe_object_fill

        err_fmag_fill = np.ones((N,))*(pbound+0.1) # should be higher than pbound
        err_fmag_fill[N // 2] = 4.0# except for this one!!
        err_fmag[:] = err_fmag_fill

        exit_wave_fill = np.array(
            [ix ** 2 + 1j * ix for ix in range(20, 20 + np.prod(backpropagated_solution.shape), 1)]).reshape(
            (M, A, B))
        exit_wave[:] = exit_wave_fill

        pa = np.zeros((M, 3), dtype=np.int32)  # not going to be used here
        oa = np.zeros((M, 3), dtype=np.int32)  # not going to be used here
        ea = np.array([np.array([ix, 0, 0]) for ix in range(M)])
        da = np.array([np.array([ix, 0, 0]) for ix in range(N)] * num_probe_modes * num_object_modes)
        ma = np.zeros((M, 3), dtype=np.int32)

        addr_info[:, 0, :] = pa
        addr_info[:, 1, :] = oa
        addr_info[:, 2, :] = ea
        addr_info[:, 3, :] = da
        addr_info[:, 4, :] = ma

        expected_out = np.array([[[-10.0 -10.0j, -10.0 -10.0j, -10.0 -8.0j, -10.0 -4.0j],
                                  [-10.0 + 2.0j, -10.0 +10.0j, -10.0 +20.0j, -10. +32.0j]],
                                 [[-766.0 -10.0j, -822.0 -10.00j, -880.0 -10.0j, -940.0 -10.0j],
                                  [-1002.0 -10.0j, -1066.0 -10.0j, -1132.0 -10.0j, -1200.0 -10.0j]],
                                 [[-10.0 +230.0j, -10.0 +262.0j, -10.0 +296.0j, -10.0 +332.0j],
                                  [-10.0 +370.0j, -10.0 +410.0j, -10.0 +452.0j, -10.0 +496.0j]],
                                 [[-10.0 +542.0j, -10.0 +590.0j, -10.0 +640.0j, -10.0 +692.0j],
                                  [-10.0 +746.0j, -10.0 +802.0j, -10.0 +860.0j, -10.0 +920.0j]],
                                 [[-2662.0 -10.0j, -2766.0 -10.0j, -2872.0 -10.0j,-2980.0 -10.0j],
                                  [-3090.0 -10.0j, -3202.0 -10.0j, -3316.0 -10.0j, -3432.0 -10.0j]],
                                 [[-10.0 +1550.0j, -10.0 +1630.0j, -10.0 +1712.0j, -10.0 +1796.0j],
                                  [-10.0 +1882.0j, -10.0 +1970.0j, -10.0 +2060.0j,-10.0 +2152.0j]]])

        out = con.get_difference(addr_info, alpha, backpropagated_solution, err_fmag, exit_wave, pbound,
                                 probe_object)

        np.testing.assert_allclose(expected_out, out)


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

        expected_out = np.array([[6.07364329e+12, 8.87756439e+13, 9.12644403e+12, 1.39851186e+14],
                                 [6.38221460e+23, 8.94021509e+24, 3.44468266e+23, 6.03329184e+24],
                                 [3.88072739e+18, 2.67132771e+19, 9.15414239e+18,4.41441340e+19]], dtype=FLOAT_TYPE)

        expected_ew = np.array([[[-4.24456960e+08 +3.33502272e+08j, -5.54233664e+08 +4.81765952e+08j, -7.26160640e+08 +6.74398016e+08j, -9.44673984e+08 +9.15834816e+08j],
                                 [-4.24455232e+08 +3.33500608e+08j, -5.54232768e+08 +4.81764832e+08j, -7.26159680e+08 +6.74396992e+08j, -9.44673792e+08 +9.15834752e+08j]],
                                [[-1.60761126e+09 +1.53736205e+09j, -1.97845542e+09 +1.92965094e+09j, -2.40919706e+09 +2.38405555e+09j, -2.90427264e+09 +2.90501248e+09j],
                                 [-1.60760742e+09 +1.53735821e+09j, -1.97845261e+09 +1.92964813e+09j, -2.40919450e+09 +2.38405299e+09j, -2.90427085e+09 +2.90501120e+09j]],
                                [[-8.77013248e+08 +4.54775968e+08j, -1.05411565e+09 +6.40012672e+08j, -1.27632653e+09 +8.72576064e+08j, -1.54808115e+09 +1.15690163e+09j],
                                 [-8.77010560e+08 +4.54773344e+08j, -1.05411462e+09 +6.40011584e+08j, -1.27632589e+09 +8.72575488e+08j, -1.54808205e+09 +1.15690304e+09j]],
                                [[-2.33229235e+09 +1.83610880e+09j, -2.75933594e+09 +2.27424461e+09j, -3.24923520e+09 +2.77745434e+09j, -3.80642586e+09 +3.35017344e+09j],
                                 [-2.33228800e+09 +1.83610432e+09j, -2.75933338e+09 +2.27424154e+09j, -3.24923290e+09 +2.77745203e+09j, -3.80642509e+09 +3.35017293e+09j]],
                                [[-1.32365261e+09 +3.21670784e+08j, -1.47709235e+09 +4.69934400e+08j, -1.67268250e+09 +6.62566528e+08j, -1.91485875e+09 +9.04003328e+08j],
                                 [-1.32365056e+09 +3.21669120e+08j, -1.47709133e+09 +4.69933312e+08j, -1.67268122e+09 +6.62565568e+08j, -1.91485837e+09 +9.04003328e+08j]],
                                [[-2.69611136e+09 +1.52553024e+09j, -3.09061837e+09 +1.91781939e+09j, -3.54502349e+09 +2.37222400e+09j, -4.06376115e+09 +2.89318067e+09j],
                                 [-2.69610726e+09 +1.52552653e+09j, -3.09061555e+09 +1.91781658e+09j, -3.54502042e+09 +2.37222144e+09j, -4.06375987e+09 +2.89317965e+09j]],
                                [[-2.15481728e+09 +4.42944416e+08j, -2.35558246e+09 +6.28181120e+08j, -2.60145664e+09 +8.60744448e+08j, -2.89687424e+09 +1.14507034e+09j],
                                 [-2.15481421e+09 +4.42941824e+08j, -2.35558118e+09 +6.28180096e+08j, -2.60145562e+09 +8.60743936e+08j, -2.89687475e+09 +1.14507149e+09j]],
                                [[-3.79940096e+09 +1.82427738e+09j, -4.25010765e+09 +2.26241280e+09j, -4.76367002e+09 +2.76562253e+09j, -5.34452326e+09 +3.33834163e+09j],
                                 [-3.79939610e+09 +1.82427277e+09j, -4.25010458e+09 +2.26240998e+09j, -4.76366746e+09 +2.76562022e+09j, -5.34452275e+09 +3.33834138e+09j]]], dtype= COMPLEX_TYPE)

        out = con.difference_map_fourier_constraint(mask, Idata, obj, probe, exit_wave, addr_info, prefilter, postfilter, pbound=pbound, alpha=alpha, LL_error=True, do_realspace_error=True)
        np.testing.assert_allclose(out,
                                   expected_out,
                                   err_msg="The returned errors are not consistent.")

        np.testing.assert_allclose(exit_wave,
                                   expected_ew,
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

        expected_out = np.array([[6.07364329e+12, 8.87756439e+13, 9.12644403e+12, 1.39851186e+14],
                                 [0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0]],
                                dtype=FLOAT_TYPE)

        expected_ew = np.array([[[-4.24456960e+08 + 3.33502272e+08j, -5.54233664e+08 + 4.81765952e+08j,
                                  -7.26160640e+08 + 6.74398016e+08j, -9.44673984e+08 + 9.15834816e+08j],
                                 [-4.24455232e+08 + 3.33500608e+08j, -5.54232768e+08 + 4.81764832e+08j,
                                  -7.26159680e+08 + 6.74396992e+08j, -9.44673792e+08 + 9.15834752e+08j]],
                                [[-1.60761126e+09 + 1.53736205e+09j, -1.97845542e+09 + 1.92965094e+09j,
                                  -2.40919706e+09 + 2.38405555e+09j, -2.90427264e+09 + 2.90501248e+09j],
                                 [-1.60760742e+09 + 1.53735821e+09j, -1.97845261e+09 + 1.92964813e+09j,
                                  -2.40919450e+09 + 2.38405299e+09j, -2.90427085e+09 + 2.90501120e+09j]],
                                [[-8.77013248e+08 + 4.54775968e+08j, -1.05411565e+09 + 6.40012672e+08j,
                                  -1.27632653e+09 + 8.72576064e+08j, -1.54808115e+09 + 1.15690163e+09j],
                                 [-8.77010560e+08 + 4.54773344e+08j, -1.05411462e+09 + 6.40011584e+08j,
                                  -1.27632589e+09 + 8.72575488e+08j, -1.54808205e+09 + 1.15690304e+09j]],
                                [[-2.33229235e+09 + 1.83610880e+09j, -2.75933594e+09 + 2.27424461e+09j,
                                  -3.24923520e+09 + 2.77745434e+09j, -3.80642586e+09 + 3.35017344e+09j],
                                 [-2.33228800e+09 + 1.83610432e+09j, -2.75933338e+09 + 2.27424154e+09j,
                                  -3.24923290e+09 + 2.77745203e+09j, -3.80642509e+09 + 3.35017293e+09j]],
                                [[-1.32365261e+09 + 3.21670784e+08j, -1.47709235e+09 + 4.69934400e+08j,
                                  -1.67268250e+09 + 6.62566528e+08j, -1.91485875e+09 + 9.04003328e+08j],
                                 [-1.32365056e+09 + 3.21669120e+08j, -1.47709133e+09 + 4.69933312e+08j,
                                  -1.67268122e+09 + 6.62565568e+08j, -1.91485837e+09 + 9.04003328e+08j]],
                                [[-2.69611136e+09 + 1.52553024e+09j, -3.09061837e+09 + 1.91781939e+09j,
                                  -3.54502349e+09 + 2.37222400e+09j, -4.06376115e+09 + 2.89318067e+09j],
                                 [-2.69610726e+09 + 1.52552653e+09j, -3.09061555e+09 + 1.91781658e+09j,
                                  -3.54502042e+09 + 2.37222144e+09j, -4.06375987e+09 + 2.89317965e+09j]],
                                [[-2.15481728e+09 + 4.42944416e+08j, -2.35558246e+09 + 6.28181120e+08j,
                                  -2.60145664e+09 + 8.60744448e+08j, -2.89687424e+09 + 1.14507034e+09j],
                                 [-2.15481421e+09 + 4.42941824e+08j, -2.35558118e+09 + 6.28180096e+08j,
                                  -2.60145562e+09 + 8.60743936e+08j, -2.89687475e+09 + 1.14507149e+09j]],
                                [[-3.79940096e+09 + 1.82427738e+09j, -4.25010765e+09 + 2.26241280e+09j,
                                  -4.76367002e+09 + 2.76562253e+09j, -5.34452326e+09 + 3.33834163e+09j],
                                 [-3.79939610e+09 + 1.82427277e+09j, -4.25010458e+09 + 2.26240998e+09j,
                                  -4.76366746e+09 + 2.76562022e+09j, -5.34452275e+09 + 3.33834138e+09j]]],
                               dtype=COMPLEX_TYPE)

        out = con.difference_map_fourier_constraint(mask, Idata, obj, probe, exit_wave, addr_info, prefilter,
                                                    postfilter, pbound=pbound, alpha=alpha, LL_error=False,
                                                    do_realspace_error=False)
        np.testing.assert_allclose(out,
                                   expected_out,
                                   err_msg="The returned errors are not consistent.")

        np.testing.assert_allclose(exit_wave,
                                   expected_ew,
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
                oa[::idx,0] = idx  # multimodal could work like this, but this is not a concrete thing (less likely for object)
        ea = np.array([np.array([ix, 0, 0]) for ix in range(M)])
        da = np.array([np.array([ix, 0, 0]) for ix in range(N)] * num_probe_modes * num_object_modes)
        ma = np.zeros((M, 3), dtype=np.int32)

        addr_info[:, 0, :] = pa
        addr_info[:, 1, :] = oa
        addr_info[:, 2, :] = ea
        addr_info[:, 3, :] = da
        addr_info[:, 4, :] = ma


        expected_out = np.array([[ 0.06855128, 1.00198244, 0.10300727, 1.57845582],
                                 [6.38221460e+23, 8.94021509e+24, 3.44468266e+23, 6.03329184e+24],
                                 [1.89878600e+07, 3.28113995e+19, 4.72013640e+07, 4.89360300e+19]], dtype=FLOAT_TYPE)

        expected_ew = np.array([[[0.00000000e+00 + 0.00000000e+00j, 0.00000000e+00 + 3.80000000e+01j, -4.00000000e+01 + 1.20000000e+02j, - 1.26000000e+02 + 2.52000000e+02j],
                                 [-6.60000000e+02 + 9.24000000e+02j, - 9.66000000e+02 + 1.28800000e+03j, -1.34400000e+03 + 1.72800000e+03j, - 1.80000000e+03 + 2.25000000e+03j]],
                                [[-6.90095424e+08 + 5.49665856e+08j, - 9.02111168e+08 + 7.77216384e+08j, -1.16220429e+09 + 1.05506266e+09j, - 1.47481139e+09 + 1.38764147e+09j],
                                 [-2.52512358e+09 + 2.52505446e+09j, - 3.05479731e+09 + 3.08208256e+09j, -3.65618765e+09 + 3.71304602e+09j, - 4.33373184e+09 + 4.42238157e+09j]],
                                [[0.00000000e+00 + 3.60000000e+01j, - 3.80000000e+01 + 1.14000000e+02j, -1.20000000e+02 + 2.40000000e+02j, - 2.52000000e+02 + 4.20000000e+02j],
                                 [-9.24000000e+02 + 1.23200000e+03j, - 1.28800000e+03 + 1.65600000e+03j, -1.72800000e+03 + 2.16000000e+03j, - 2.25000000e+03 + 2.75000000e+03j]],
                                [[-1.49062272e+09 + 9.55004032e+08j, - 1.78523661e+09 + 1.25600128e+09j, -2.13328819e+09 + 1.61265498e+09j, - 2.53921434e+09 + 2.02940147e+09j],
                                 [-3.17395840e+09 + 2.71720909e+09j, - 3.73343309e+09 + 3.29248486e+09j, -4.36517990e+09 + 3.94225101e+09j, - 5.07363635e+09 + 4.67094528e+09j]],
                                [[0.00000000e+00 + 0.00000000e+00j, 0.00000000e+00 + 3.80000000e+01j, -4.00000000e+01 + 1.20000000e+02j, - 1.26000000e+02 + 2.52000000e+02j],
                                 [-6.60000000e+02 + 9.24000000e+02j, - 9.66000000e+02 + 1.28800000e+03j, -1.34400000e+03 + 1.72800000e+03j, - 1.80000000e+03 + 2.25000000e+03j]],
                                [[-1.73131635e+09 + 5.37834304e+08j, - 1.96699507e+09 + 7.65384832e+08j, -2.25075123e+09 + 1.04323117e+09j, - 2.58702131e+09 + 1.37581005e+09j],
                                 [-3.66090240e+09 + 2.51322291e+09j, - 4.21423872e+09 + 3.07025101e+09j, -4.83929190e+09 + 3.70121421e+09j, - 5.54049946e+09 + 4.41055027e+09j]],
                                [[0.00000000e+00 + 3.60000000e+01j, - 3.80000000e+01 + 1.14000000e+02j, -1.20000000e+02 + 2.40000000e+02j, - 2.52000000e+02 + 4.20000000e+02j],
                                 [-9.24000000e+02 + 1.23200000e+03j, - 1.28800000e+03 + 1.65600000e+03j, -1.72800000e+03 + 2.16000000e+03j, - 2.25000000e+03 + 2.75000000e+03j]],
                                [[-2.92006195e+09 + 9.43172416e+08j, - 3.23833907e+09 + 1.24416986e+09j, -3.61005363e+09 + 1.60082368e+09j, - 4.03964314e+09 + 2.01757018e+09j],
                                 [-4.67873485e+09 + 2.70537754e+09j, - 5.26187366e+09 + 3.28065306e+09j, -5.91728384e+09 + 3.93041971e+09j, - 6.64940288e+09 + 4.65911398e+09j]]]
                               ,dtype=COMPLEX_TYPE)

        out = con.difference_map_fourier_constraint(mask, Idata, obj, probe, exit_wave, addr_info, prefilter,
                                                    postfilter, pbound=pbound, alpha=alpha, LL_error=True,
                                                    do_realspace_error=True)

        np.testing.assert_allclose(out,
                                   expected_out,
                                   err_msg="The returned errors are not consistent.")

        np.testing.assert_allclose(exit_wave,
                                   expected_ew,
                                   err_msg="The expected in-place update of the exit wave didn't work properly.")


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

        expected_probe = np.array([[[-361.18814087-1000.74768066j, -148.70419312 +206.73210144j,
                                     -91.97548676   -9.23460865j,   16.05268097  -41.11487198j],
                                    [-152.72857666 +109.68946838j,  -62.70831680  -58.41201782j,
                                     37.14255524  -17.69169426j,   -4.25983000   -6.36473894j]],

                                   [[-749.64007568-2050.26147461j, -401.52606201 +451.15054321j,
                                     -218.26188660  -36.93523788j,   41.09194946  -91.68986511j],
                                    [-350.60354614 +226.57118225j, -109.65759277 -124.06006622j,
                                     68.98976898  -24.72795677j,   -5.64832449  -11.25561905j]]], dtype=COMPLEX_TYPE)

        expected_obj = np.array([[[ -0.00000000e+00 -0.00000000e+00j,  -1.27605135e-02 -1.07031791e-02j,
                                    1.80982396e-01 -1.77402645e-01j,  -2.30214819e-01 -1.09420657e+00j,
                                    -5.07897234e+00 -9.75304246e-01j,   5.00000000e+00 +2.50000000e+01j],
                                  [ -1.48292825e-01 -4.61030722e-01j,  -5.86787999e-01 +3.41154838e+00j,
                                    -1.08777246e+01 -8.06874752e+00j,  -3.86462593e+01 +1.36726942e+01j,
                                    6.36259308e+01 +8.40691147e+01j,   1.10000000e+01 +1.21000000e+02j],
                                  [  1.10468502e+01 -5.40999889e+00j,  -2.75495262e+01 -7.16127729e+00j,
                                     -3.61000290e+01 +9.38624268e+01j,   2.70963776e+02 -2.03708401e+01j,
                                     2.82062347e+02 +2.01701343e+03j,   1.70000000e+01 +2.89000000e+02j],
                                  [  1.80000000e+01 +3.24000000e+02j,   1.90000000e+01 +3.61000000e+02j,
                                     2.00000000e+01 +4.00000000e+02j,   2.10000000e+01 +4.41000000e+02j,
                                     2.20000000e+01 +4.84000000e+02j,   2.30000000e+01 +5.29000000e+02j]]], dtype=COMPLEX_TYPE)

        expected_errors = np.array([[[  1.30852982e-01+0.j,   7.86592126e-01+0.j,   2.91434258e-01+0.j,
                                        1.26918125e+00+0.j],
                                     [  2.88552762e+24+0.j,   6.12232725e+25+0.j,  6.29929433e+23+0.j,
                                        4.21546840e+25+0.j],
                                     [  1.75457960e+07+0.j,   7.23184240e+07+0.j,   4.43651240e+07+0.j,
                                        3.27585548e+19+0.j]],

                                    [[  1.28861861e-02+0.j,   1.27825022e-01+0.j,   2.06416398e-02+0.j,
                                        1.36703640e+11+0.j],
                                     [  2.88552762e+24+0.j,   6.12232725e+25+0.j,   6.29929433e+23+0.j,
                                        4.21546840e+25+0.j],
                                     [  0.00000000e+00+0.j,   0.00000000e+00+0.j,   0.00000000e+00+0.j,
                                        3.27586977e+19+0.j]]], dtype=COMPLEX_TYPE)




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



        np.testing.assert_array_equal(expected_probe,
                                      probe,
                                      err_msg="The probe has not behaved as expected.")

        np.testing.assert_array_equal(expected_obj,
                                      obj,
                                      err_msg="The object has not behaved as expected.")

        np.testing.assert_array_equal(expected_errors,
                                      errors,
                                      err_msg="The errors have not behaved as expected.")

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

        expected_probe = deepcopy(probe)

        expected_obj = np.array([[[ -0.00000000e+00 -0.00000000e+00j,
                                    1.00000000e+00 +1.00000000e+00j,
                                    2.00000000e+00 +4.00000000e+00j,
                                    3.00000000e+00 +9.00000000e+00j,
                                    4.00000000e+00 +1.60000000e+01j,
                                    5.00000000e+00 +2.50000000e+01j],
                                  [  6.00000000e+00 +3.60000000e+01j,
                                     -7.95981900e+06 +1.43803590e+07j,
                                     -7.64522750e+06 +1.61365070e+07j,
                                     -7.34606000e+06 +1.82074500e+07j,
                                     -1.48699840e+07 +4.33746560e+07j,
                                     1.10000000e+01 +1.21000000e+02j],
                                  [  1.20000000e+01 +1.44000000e+02j,
                                     -1.60912900e+07 +7.23797920e+07j,
                                     -1.52878190e+07 +8.04868400e+07j,
                                     -1.45362140e+07 +8.92972240e+07j,
                                     -2.90441140e+07 +2.07500928e+08j,
                                     1.70000000e+01 +2.89000000e+02j],
                                  [  1.80000000e+01 +3.24000000e+02j,
                                     1.90000000e+01 +3.61000000e+02j,
                                     2.00000000e+01 +4.00000000e+02j,
                                     2.10000000e+01 +4.41000000e+02j,
                                     2.20000000e+01 +4.84000000e+02j,
                                     2.30000000e+01 +5.29000000e+02j]]], dtype=COMPLEX_TYPE)

        expected_errors = np.array([[[  1.30852982e-01+0.j,   7.86592126e-01+0.j,   2.91434258e-01+0.j,
                                              1.26918125e+00+0.j],
                                           [  2.88552762e+24+0.j,   6.12232725e+25+0.j,   6.29929433e+23+0.j,
                                              4.21546840e+25+0.j],
                                           [  1.75457960e+07+0.j,   7.23184240e+07+0.j,   4.43651240e+07+0.j,
                                              3.27585548e+19+0.j]]], dtype=COMPLEX_TYPE)


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
                                             probe_update_start=2,
                                             alpha=alpha,
                                             clip_object=None,
                                             LL_error=True,
                                             num_iterations=num_iter)

        np.testing.assert_array_equal(expected_probe,
                                      probe,
                                      err_msg="The probe has not behaved as expected.")

        np.testing.assert_array_equal(expected_obj,
                                      obj,
                                      err_msg="The object has not behaved as expected.")

        np.testing.assert_array_equal(expected_errors,
                                      errors,
                                      err_msg="The error has not behaved as expected.")



if __name__ == '__main__':
    unittest.main()



