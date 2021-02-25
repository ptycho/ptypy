'''
The tests for the constraints
'''

import unittest
import numpy as np
from copy import deepcopy
from ptypy.accelerate.array_based import constraints as con, FLOAT_TYPE, COMPLEX_TYPE
from . import have_cuda, only_if_cuda_available

if have_cuda():
    from archive.cuda_extension.accelerate.cuda import constraints as gcon

@only_if_cuda_available
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


    def test_get_difference_pbound_is_none(self):
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
            [ix + 1j * ix for ix in range(10, 10 + np.prod(backpropagated_solution.shape), 1)]).reshape(
            (M, A, B))
        probe_object[:] = probe_object_fill

        err_fmag_fill = np.ones((N,))
        err_fmag[:] = err_fmag_fill  # this shouldn't be used as pbound is None

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

        gout = gcon.get_difference(addr_info, alpha, backpropagated_solution, err_fmag, exit_wave, pbound,
                                 probe_object)

        out = con.get_difference(addr_info, alpha, backpropagated_solution, err_fmag, exit_wave, pbound,
                                 probe_object)
        out_test = out.reshape((np.prod(out.shape),))
        gout_test = gout.reshape((np.prod(gout.shape),))
        for idx in range(len(out)):
            np.testing.assert_allclose(out_test[idx],
                                       gout_test[idx],
                                       err_msg=("failed on index:%s\n" % (idx)))

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
            [ix + 1j * ix for ix in range(10, 10 + np.prod(backpropagated_solution.shape), 1)]).reshape(
            (M, A, B))
        probe_object[:] = probe_object_fill

        err_fmag_fill = np.ones((N,)) * (pbound + 0.1)  # should be higher than pbound
        err_fmag_fill[N // 2] = 4.0  # except for this one!!
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

        gout = gcon.get_difference(addr_info, alpha, backpropagated_solution, err_fmag, exit_wave, pbound,
                                 probe_object)

        out = con.get_difference(addr_info, alpha, backpropagated_solution, err_fmag, exit_wave, pbound,
                                 probe_object)
        out_test = out.reshape((np.prod(out.shape),))
        gout_test = gout.reshape((np.prod(gout.shape),))

        for idx in range(len(out)):
            np.testing.assert_allclose(out_test[idx],
                                       gout_test[idx],
                                       err_msg=("failed on index:%s\n" % (idx)))

    def test_difference_map_fourier_constraint_pbound_is_none_with_realspace_error_and_LL_error(self):

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

        out = con.difference_map_fourier_constraint(mask, Idata, obj, probe, exit_wave, addr_info, prefilter,
                                                    postfilter, pbound=pbound, alpha=alpha, LL_error=True,
                                                    do_realspace_error=True)

        gout = gcon.difference_map_fourier_constraint(mask, Idata, obj, probe, gexit_wave, addr_info, prefilter,
                                                    postfilter, pbound=pbound, alpha=alpha, LL_error=True,
                                                    do_realspace_error=True)
        np.testing.assert_allclose(out,
                                   gout,
                                   rtol=1e-6,
                                   err_msg="The returned errors are not consistent.")

        np.testing.assert_allclose(exit_wave,
                                   gexit_wave,
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

        out = con.difference_map_fourier_constraint(mask, Idata, obj, probe, exit_wave, addr_info, prefilter,
                                                    postfilter, pbound=pbound, alpha=alpha, LL_error=False,
                                                    do_realspace_error=False)

        gout = con.difference_map_fourier_constraint(mask, Idata, obj, probe, gexit_wave, addr_info, prefilter,
                                                    postfilter, pbound=pbound, alpha=alpha, LL_error=False,
                                                    do_realspace_error=False)

        np.testing.assert_allclose(out,
                                   gout,
                                   err_msg="The returned errors are not consistent.")

        np.testing.assert_allclose(exit_wave,
                                   gexit_wave,
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

        out = con.difference_map_fourier_constraint(mask, Idata, obj, probe, exit_wave, addr_info, prefilter,
                                                    postfilter, pbound=pbound, alpha=alpha, LL_error=True,
                                                    do_realspace_error=True)
        gout = gcon.difference_map_fourier_constraint(mask, Idata, obj, probe, gexit_wave, addr_info, prefilter,
                                                    postfilter, pbound=pbound, alpha=alpha, LL_error=True,
                                                    do_realspace_error=True)
        np.testing.assert_allclose(out,
                                   gout,
                                   rtol=1e-6,
                                   err_msg="The returned errors are not consistent.")

        np.testing.assert_allclose(exit_wave,
                                   gexit_wave,
                                   rtol=1e-6,
                                   err_msg="The expected in-place update of the exit wave didn't work properly.")


if __name__ == '__main__':
    unittest.main()



