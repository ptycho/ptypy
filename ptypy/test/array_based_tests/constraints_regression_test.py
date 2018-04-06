'''
The tests for the constraints
'''


import unittest
import numpy as np
from ptypy.array_based import constraints as con, FLOAT_TYPE, COMPLEX_TYPE

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
        np.testing.assert_allclose(out, expected_out)


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

    # def difference_map_fourier_constraint(mask, Idata, obj, probe, exit_wave, addr_info, prefilter, postfilter,
    #                                       pbound=None, alpha=1.0, LL_error=True, do_realspace_error=True):
    #     '''
    #     This kernel just performs the fourier renormalisation.
    #     :param mask. The nd mask array
    #     :param diffraction. The nd diffraction data
    #     :param farfield_stack. The current iterant.
    #     :param addr. The addresses of the stacks.
    #     :return: The updated iterant
    #             : fourier errors
    #     '''
    #
    #     probe_object = scan_and_multiply(probe, obj, exit_wave.shape, addr_info)
    #
    #     # Buffer for accumulated photons
    #     # For log likelihood error # need to double check this adp
    #     if LL_error is True:
    #         err_phot = log_likelihood(probe_object, mask, Idata, prefilter, postfilter, addr_info)
    #     else:
    #         err_phot = np.zeros(Idata.shape[0], dtype=FLOAT_TYPE)
    #
    #     constrained = difference_map_realspace_constraint(probe_object, exit_wave, alpha)
    #     f = farfield_propagator(constrained, prefilter, postfilter, direction='forward')
    #     pa, oa, ea, da, ma = zip(*addr_info)
    #     af2 = au.sum_to_buffer(au.abs2(f), Idata.shape, ea, da, dtype=FLOAT_TYPE)
    #
    #     fmag = np.sqrt(np.abs(Idata))
    #     af = np.sqrt(af2)
    #     # # Fourier magnitudes deviations(current_solution, pbound, measured_solution, mask, addr)
    #     err_fmag = far_field_error(af, fmag, mask)
    #
    #     vectorised_rfm = renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound)
    #
    #     backpropagated_solution = farfield_propagator(vectorised_rfm,
    #                                                   postfilter.conj(),
    #                                                   prefilter.conj(),
    #                                                   direction='backward')
    #
    #     df = get_difference(addr_info, alpha, backpropagated_solution, err_fmag, exit_wave, pbound, probe_object)
    #
    #     exit_wave += df
    #     if do_realspace_error:
    #         ea_first_column = np.array(ea)[:, 0]
    #         da_first_column = np.array(da)[:, 0]
    #         err_exit = realspace_error(df, ea_first_column, da_first_column, Idata.shape[0])
    #     else:
    #         err_exit = np.zeros((Idata.shape[0]))
    #
    #     if pbound is not None:
    #         err_fmag /= pbound
    #
    #     return np.array([err_fmag, err_phot, err_exit])


if __name__ == '__main__':
    unittest.main()



