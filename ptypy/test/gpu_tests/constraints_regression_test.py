'''
The tests for the constraints
'''


import unittest
import numpy as np
from ptypy.array_based import constraints as con, FLOAT_TYPE, COMPLEX_TYPE
from ptypy.gpu import constraints as gcon

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

        out = con.renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound)
        gout = gcon.renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound)

        out_test = out.reshape((np.prod(out.shape),))
        gout_test = gout.reshape((np.prod(gout.shape),))
        for idx in range(len(out)):
            np.testing.assert_allclose(out_test[idx],
                                       gout_test[idx],
                                       err_msg=("failed on index:%s\n" % (idx)))


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

        gout = gcon.renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound)
        out = con.renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound)
        out_test = out.reshape((np.prod(out.shape),))
        gout_test = gout.reshape((np.prod(gout.shape),))
        for idx in range(len(out)):
            np.testing.assert_allclose(out_test[idx],
                                       gout_test[idx],
                                       err_msg=("failed on index:%s\n" % (idx)))

    #
    # def get_difference(addr_info, alpha, backpropagated_solution, err_fmag, exit_wave, pbound, probe_object):
    #     df = np.zeros(exit_wave.shape, dtype=np.complex128)
    #     for _pa, _oa, ea, da, ma in addr_info:
    #         if (pbound is None) or (err_fmag[da[0]] > pbound):
    #             df[ea[0]] = np.subtract(backpropagated_solution[ea[0]], probe_object[ea[0]])
    #         else:
    #             df[ea[0]] = alpha * np.subtract(probe_object[ea[0]], exit_wave[ea[0]])
    #     return df
    #
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



