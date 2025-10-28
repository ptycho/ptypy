'''
Wrapper of the CUDA extensions

Contains all cython wrappers to the CUDA extension module,
to allow a single library build and interactions instead of many
'''

import numpy as np
from . import COMPLEX_TYPE
from cuda_functions cimport *
cimport numpy as np 
import cython
import time
from libcpp.string cimport string

@cython.boundscheck(False)
@cython.wraparound(False)
def difference_map_realspace_constraint(obj_and_probe, exit_wave, alpha):
    cdef np.complex64_t [:,:,::1] obj_and_probe_c = np.ascontiguousarray(obj_and_probe)
    cdef np.complex64_t [:,:,::1] exit_wave_c = np.ascontiguousarray(exit_wave)
    cdef float alpha_c = alpha
    out = np.empty(exit_wave.shape, dtype=np.complex64, order='C')
    cdef np.complex64_t [:,:,::1] out_c = out
    difference_map_realspace_constraint_c(
        <const float*>&obj_and_probe_c[0,0,0],
        <const float*>&exit_wave_c[0,0,0],
        alpha_c,
        obj_and_probe.shape[0],
        obj_and_probe.shape[1],
        obj_and_probe.shape[2],
        <float*>&out_c[0,0,0]
    )
    return out



@cython.boundscheck(False)
@cython.wraparound(False)
def scan_and_multiply(probe, obj, exit_shape, addresses):
    cdef np.complex64_t [:,:,::1] probe_c = np.ascontiguousarray(probe)
    cdef np.complex64_t [:,:,::1] obj_c = np.ascontiguousarray(obj)
    cdef i_probe = probe.shape[0]
    cdef m_probe = probe.shape[1]
    cdef n_probe = probe.shape[2]
    cdef i_obj = obj.shape[0]
    cdef m_obj = obj.shape[1]
    cdef n_obj = obj.shape[2]
    cdef np.int32_t [:,:,::1] addr_info_c = np.ascontiguousarray(addresses)
    cdef int addr_len = addresses.shape[0]
    cdef batch_size = exit_shape[0]
    cdef m = exit_shape[1]
    cdef n = exit_shape[2]
    out = np.empty(exit_shape, dtype=np.complex64, order='C')
    cdef np.complex64_t [:,:,::1] out_c = out
    scan_and_multiply_c(
        <const float*>&probe_c[0,0,0],
        i_probe, m_probe, n_probe,
        <const float*>&obj_c[0,0,0],
        i_obj, m_obj, n_obj,
        <int*>&addr_info_c[0,0,0],
        addr_len,
        batch_size, m, n,
        <float*>&out_c[0,0,0]
    )
    return out

@cython.wraparound(False)
@cython.boundscheck(False)
def farfield_propagator(
        data_to_be_transformed not None, 
        prefilter=None, 
        postfilter=None, 
        direction='forward'
    ):
    cdef np.complex64_t [:,:,::1] x = np.ascontiguousarray(data_to_be_transformed)
    out = np.empty_like(x)
    cdef np.complex64_t [:,:,::1] out_c = out
    
    dtype = np.complex64

    cdef np.complex64_t [:,::1] prefilter_c = None
    if not prefilter is None:
        prefilter_c = np.ascontiguousarray(prefilter.astype(dtype))
    
    cdef np.complex64_t [:,::1] postfilter_c = None
    if not postfilter is None:
        postfilter_c = np.ascontiguousarray(postfilter.astype(dtype))

    cdef b = data_to_be_transformed.shape[0]
    cdef m = data_to_be_transformed.shape[1]
    cdef n = data_to_be_transformed.shape[2]
    cdef isForward = 0
    if direction == 'forward':
        isForward = 1
    farfield_propagator_c(
        <const float*>&x[0,0,0],
        <const float*>&prefilter_c[0,0],
        <const float*>&postfilter_c[0,0],
        <float*>&out_c[0,0,0],
        b, m, n, isForward
    )   
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def sqrt_abs(np.complex64_t [:,::1] diffraction not None):
    out = np.empty_like(diffraction)
    cdef np.complex64_t[:,::1] out_c = out
    cdef m = diffraction.shape[0]
    cdef n = diffraction.shape[1]
    sqrt_abs_c(<const float*>&diffraction[0,0], <float*>&out_c[0,0], m, n)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def log_likelihood(probe_obj, mask, Idata, prefilter, postfilter, addr_info):
    cdef np.complex64_t [:,:,::1] probe_obj_c = np.ascontiguousarray(probe_obj)
    cdef np.uint8_t [::1] mask_c = np.frombuffer(np.ascontiguousarray(mask.astype(np.bool)), dtype=np.uint8)
    cdef np.float32_t [:,:,::1] Idata_c = np.ascontiguousarray(Idata)
    cdef np.complex64_t [:,::1] prefilter_c = np.ascontiguousarray(prefilter)
    cdef np.complex64_t [:,::1] postfilter_c = np.ascontiguousarray(postfilter)
    cdef np.int32_t [:,:,::1] addr_info_c = np.ascontiguousarray(addr_info)
    out = np.empty(Idata.shape[0], np.float32)
    cdef np.float32_t [::1] out_c = out
    cdef int i = probe_obj.shape[0]
    cdef int m = probe_obj.shape[1]
    cdef int n = probe_obj.shape[2] 
    log_likelihood_c(
        <const float*>&probe_obj_c[0,0,0],
        <const unsigned char*>&mask_c[0],
        <const float*>&Idata_c[0,0,0],
        <const float*>&prefilter_c[0,0],
        <const float*>&postfilter_c[0,0],
        <const int*>&addr_info_c[0,0,0],
        <float*>&out_c[0],
        i, m, n, addr_info.shape[0],
        Idata.shape[0]
    )
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def abs2(input):
    cin = np.ascontiguousarray(input)
    outtype = cin.dtype
    if cin.dtype == np.complex64:
        outtype = np.float32
    elif cin.dtype == np.complex128:
        outtype = np.float64
    cout = np.empty(cin.shape, dtype=outtype)
    cdef np.float32_t [:,::1] cout_2c
    cdef np.float32_t [:,:,::1] cout_3c
    cdef np.float64_t [:,::1] cout_d2c
    cdef np.float64_t [:,:,::1] cout_d3c
    cdef int n = np.prod(cin.shape)

    cdef np.float32_t [:, ::1] cin_f2c
    cdef np.complex64_t [:, ::1] cin_c2c
    cdef np.float32_t [:,:, ::1] cin_f3c
    cdef np.complex64_t [:,:, ::1] cin_c3c
    cdef np.float64_t [:, ::1] cin_d2c
    cdef np.complex128_t [:, ::1] cin_z2c
    cdef np.float64_t [:,:, ::1] cin_d3c
    cdef np.complex128_t [:,:, ::1] cin_z3c

    if len(cin.shape) == 2:
        if (cin.dtype == np.float32):
            cout_2c = cout
            cin_f2c = cin
            abs2_c(<const float*>&cin_f2c[0,0], <float*>&cout_2c[0,0], n, 0)
        elif (cin.dtype == np.complex64):
            cout_2c = cout
            cin_c2c = cin
            abs2_c(<const float*>&cin_c2c[0,0], <float*>&cout_2c[0,0], n, 1)
        elif (cin.dtype == np.float64):
            cout_d2c = cout
            cin_d2c = cin
            abs2d_c(<const double*>&cin_d2c[0,0], <double*>&cout_d2c[0,0], n, 0)
        elif (cin.dtype == np.complex128):
            cout_d2c = cout
            cin_z2c = cin
            abs2d_c(<const double*>&cin_z2c[0,0], <double*>&cout_d2c[0,0], n, 1)
        else:
            raise ValueError("unsupported datatype {}".format(cin.dtype))
    elif len(cin.shape) == 3:
        if (cin.dtype == np.float32):
            cout_3c = cout
            cin_f3c = cin
            abs2_c(<const float*>&cin_f3c[0,0,0], <float*>&cout_3c[0,0,0], n, 0)
        elif (cin.dtype == np.complex64):
            cout_3c = cout
            cin_c3c = cin
            abs2_c(<const float*>&cin_c3c[0,0,0], <float*>&cout_3c[0,0,0], n, 1)
        elif (cin.dtype == np.float64):
            cout_d3c = cout
            cin_d3c = cin
            abs2d_c(<const double*>&cin_d3c[0,0,0], <double*>&cout_d3c[0,0,0], n, 0)
        elif (cin.dtype == np.complex128):
            cout_d3c = cout
            cin_z3c = cin
            abs2d_c(<const double*>&cin_z3c[0,0,0], <double*>&cout_d3c[0,0,0], n, 1)
        else:
            raise ValueError("unsupported datatype {}".format(cin.dtype))
    else:
        raise ValueError("unsupported dimensionality: {}".format(len(cin.shape)))
    return cout

@cython.boundscheck(False)
@cython.wraparound(False)
def _sum_to_buffer_real(in1, outshape, in1_addr, out1_addr):
    cdef np.float32_t [:,:,::1] in1_c = np.ascontiguousarray(in1)
    cdef int os_0 = outshape[0]
    cdef int os_1 = outshape[1]
    cdef int os_2 = outshape[2]
    cdef np.int32_t [:,:] in1_addr_c = np.ascontiguousarray(in1_addr)
    cdef np.int32_t [:,:] out1_addr_c = np.ascontiguousarray(out1_addr)
    out = np.empty(outshape, dtype=np.float32)
    cdef np.float32_t [:,:,::1] out_c = out
    # dimensions
    cdef int in1_0 = in1.shape[0]
    cdef int in1_1 = in1.shape[1]
    cdef int in1_2 = in1.shape[2]
    cdef int in_addr_0 = in1_addr.shape[0]
    cdef int out_addr_0 = out1_addr.shape[0]
    sum_to_buffer_c(
        <const float*>&in1_c[0,0,0],
        in1_0, in1_1, in1_2,
        <float*>&out_c[0,0,0],
        os_0, os_1, os_2,
        <const int*>&in1_addr_c[0,0],
        in_addr_0, 
        <const int*>&out1_addr_c[0,0],
        out_addr_0,
        0
    )
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def _sum_to_buffer_stride_real(in1, outshape, addr_info):
    cdef np.float32_t [:,:,::1] in1_c = np.ascontiguousarray(in1)
    cdef int os_0 = outshape[0]
    cdef int os_1 = outshape[1]
    cdef int os_2 = outshape[2]
    cdef np.int32_t [:,:,::1] addr_info_c = np.ascontiguousarray(addr_info)
    out = np.empty(outshape, dtype=np.float32)
    cdef np.float32_t [:,:,::1] out_c = out
    # dimensions
    cdef int in1_0 = in1.shape[0]
    cdef int in1_1 = in1.shape[1]
    cdef int in1_2 = in1.shape[2]
    cdef int addr_info_0 = addr_info.shape[0]
    sum_to_buffer_stride_c(
        <const float*>&in1_c[0,0,0],
        in1_0, in1_1, in1_2,
        <float*>&out_c[0,0,0],
        os_0, os_1, os_2,
        <const int*>&addr_info_c[0,0,0],
        addr_info_0, 
        0
    )
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def _sum_to_buffer_complex(in1, outshape, in1_addr, out1_addr):
    cdef np.complex64_t [:,:,::1] in1_c = np.ascontiguousarray(in1)
    cdef int os_0 = outshape[0]
    cdef int os_1 = outshape[1]
    cdef int os_2 = outshape[2]
    cdef np.int32_t [:,:] in1_addr_c = np.ascontiguousarray(in1_addr)
    cdef np.int32_t [:,:] out1_addr_c = np.ascontiguousarray(out1_addr)
    out = np.empty(outshape, dtype=np.complex64)
    cdef np.complex64_t [:,:,::1] out_c = out
    # dimensions
    cdef int in1_0 = in1.shape[0]
    cdef int in1_1 = in1.shape[1]
    cdef int in1_2 = in1.shape[2]
    cdef int in_addr_0 = in1_addr.shape[0]
    cdef int out_addr_0 = out1_addr.shape[0]
    sum_to_buffer_c(
        <const float*>&in1_c[0,0,0],
        in1_0, in1_1, in1_2,
        <float*>&out_c[0,0,0],
        os_0, os_1, os_2,
        <const int*>&in1_addr_c[0,0],
        in_addr_0,
        <const int*>&out1_addr_c[0,0],
        out_addr_0,
        1
    )
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def _sum_to_buffer_stride_complex(in1, outshape, addr_info):
    cdef np.complex64_t [:,:,::1] in1_c = np.ascontiguousarray(in1)
    cdef int os_0 = outshape[0]
    cdef int os_1 = outshape[1]
    cdef int os_2 = outshape[2]
    cdef np.int32_t [:,:,::1] addr_info_c = np.ascontiguousarray(addr_info)
    out = np.empty(outshape, dtype=np.complex64)
    cdef np.complex64_t [:,:,::1] out_c = out
    # dimensions
    cdef int in1_0 = in1.shape[0]
    cdef int in1_1 = in1.shape[1]
    cdef int in1_2 = in1.shape[2]
    cdef int addr_info_0 = addr_info.shape[0]
    sum_to_buffer_stride_c(
        <const float*>&in1_c[0,0,0],
        in1_0, in1_1, in1_2,
        <float*>&out_c[0,0,0],
        os_0, os_1, os_2,
        <const int*>&addr_info_c[0,0,0],
        addr_info_0,
        1
    )
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_to_buffer(in1, outshape, in1_addr, out1_addr, dtype):
    if not isinstance(in1_addr, np.ndarray):
        in1_addr = np.array(in1_addr, dtype=np.int32)
    if not isinstance(out1_addr, np.ndarray):
        out1_addr = np.array(out1_addr, dtype=np.int32)
    if dtype == np.float32:
        return _sum_to_buffer_real(in1, outshape, in1_addr.astype(np.int32), out1_addr.astype(np.int32))
    elif dtype == np.complex64:
        return _sum_to_buffer_complex(in1, outshape, in1_addr.astype(np.int32), out1_addr.astype(np.int32))

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_to_buffer_stride(in1, outshape, addr_info, dtype):
    if not isinstance(addr_info, np.ndarray):
        addr_info = np.array(addr_info, dtype=np.int32)
    if dtype == np.float32:
        return _sum_to_buffer_stride_real(in1, outshape, addr_info.astype(np.int32))
    elif dtype == np.complex64:
        return _sum_to_buffer_stride_complex(in1, outshape, addr_info.astype(np.int32))


@cython.boundscheck(False)
@cython.wraparound(False)
def far_field_error(current_solution, measured_solution, mask):
    cdef np.float32_t [:,:,::1] current_c = np.ascontiguousarray(current_solution)
    cdef np.float32_t [:,:,::1] measured_c = np.ascontiguousarray(measured_solution)
    cdef np.uint8_t [::1] mask_c = np.frombuffer(np.ascontiguousarray(mask.astype(np.bool)), dtype=np.uint8)
    out = np.empty((current_c.shape[0],), np.float32)
    cdef np.float32_t [::1] out_c = out
    cdef i = current_solution.shape[0]
    cdef m = current_solution.shape[1]
    cdef n = current_solution.shape[2]
    far_field_error_c(
        <const float*>&current_c[0,0,0],
        <const float*>&measured_c[0,0,0],
        <const unsigned char*>&mask_c[0],
        <float*>&out_c[0],
        i, m, n
    )
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def realspace_error(difference, ea_first_column, da_first_column, out_length):
    cdef np.complex64_t [:,:,::1] difference_c = np.ascontiguousarray(difference)
    out = np.empty((out_length,), dtype=np.float32)
    cdef np.float32_t [::1] out_c = out
    cdef i = difference.shape[0]
    cdef m = difference.shape[1]
    cdef n = difference.shape[2]
    if not isinstance(ea_first_column, np.ndarray):
        ea_first_column = np.array(ea_first_column, dtype=np.int32)
    if not isinstance(da_first_column, np.ndarray):
        da_first_column = np.array(da_first_column, dtype=np.int32)
    cdef np.int32_t [::1] ea_first_column_c = np.ascontiguousarray(ea_first_column)
    cdef np.int32_t [::1] da_first_column_c = np.ascontiguousarray(da_first_column)
    cdef int addr_len = min(ea_first_column.shape[0], da_first_column.shape[0])
    cdef int out_length_c = out_length
    realspace_error_c(
        <const float*>&difference_c[0,0,0],
        <const int*>&ea_first_column_c[0],
        <const int*>&da_first_column_c[0],
        addr_len,
        <float*>&out_c[0],
        i, m, n,
        out_length_c
    )
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def get_difference(addr_info, alpha, backpropagated_solution, err_fmag, exit_wave, pbound, probe_object):
    cdef np.int32_t [:,:,::1] addr_info_c = np.ascontiguousarray(addr_info)
    cdef alpha_c = alpha
    bp_sol = backpropagated_solution.astype(np.complex64)
    cdef np.complex64_t [:,:,::1] bp_sol_c = np.ascontiguousarray(bp_sol)
    err_fmag_f32 = err_fmag.astype(np.float32)
    cdef np.float32_t [::1] err_fmag_c = np.ascontiguousarray(err_fmag_f32)
    cdef np.complex64_t [:,:,::1] exit_wave_c = np.ascontiguousarray(exit_wave)
    cdef pbound_c = 0.0
    if not pbound is None:
        pbound_c = pbound
    cdef np.complex64_t [:,:,::1] probe_obj_c = np.ascontiguousarray(probe_object)
    out = np.empty_like(exit_wave)
    cdef np.complex64_t [:,:,::1] out_c = out
    cdef int i = exit_wave.shape[0]
    cdef int m = exit_wave.shape[1]
    cdef int n = exit_wave.shape[2]
    get_difference_c(
        <const int*>&addr_info_c[0,0,0],
        alpha_c,
        <const float*>&bp_sol_c[0,0,0],
        <const float*>&err_fmag_c[0],
        <const float*>&exit_wave_c[0,0,0],
        pbound_c,
        <const float*>&probe_obj_c[0,0,0],
        <float*>&out_c[0,0,0],
        i, m, n,
        0 if pbound == None else 1
    )
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound):
    cdef np.complex64_t [:,:,::1] f_c = np.ascontiguousarray(f)
    cdef np.float32_t [:,:,::1] af_c = np.ascontiguousarray(af)
    cdef np.float32_t [:, :, ::1] fmag_c = np.ascontiguousarray(fmag)
    cdef np.uint8_t [::1] mask_c = np.frombuffer(np.ascontiguousarray(mask.astype(np.bool)), dtype=np.uint8)
    cdef np.float32_t [::1] err_fmag_c = np.ascontiguousarray(err_fmag.astype(np.float32))
    cdef np.int32_t [:,:,::1] addr_info_c = np.ascontiguousarray(addr_info)
    cdef pbound_c = 0.0
    if not pbound is None:
        pbound_c = pbound
    out = np.empty_like(f)
    cdef np.complex64_t [:,:,::1] out_c = out
    cdef int M = f.shape[0]
    cdef int N = fmag.shape[0]
    cdef int A = f.shape[1]
    cdef int B = f.shape[2]
    assert(M == out.shape[0])
    # assert(N == af.shape[0])
    assert(N == mask.shape[0])
    assert(N == err_fmag.shape[0])
    assert(len(err_fmag.shape) == 1)
    assert(M == addr_info.shape[0])
    renormalise_fourier_magnitudes_c(
        <const float*>&f_c[0,0,0],
        <const float*>&af_c[0,0,0],
        <const float*>&fmag_c[0,0,0],
        <const unsigned char*>&mask_c[0],
        <const float*>&err_fmag_c[0],
        <const int*>&addr_info_c[0,0,0],
        pbound_c,
        <float*>&out_c[0,0,0],
        M, N, A, B,
        0 if pbound == None else 1
    )
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def difference_map_fourier_constraint(mask, Idata, obj, probe, exit_wave, addr_info, prefilter, postfilter, pbound=None, alpha=1.0, LL_error=True, do_realspace_error=True):
    cdef np.uint8_t [::1] mask_c = np.frombuffer(np.ascontiguousarray(mask.astype(np.bool)), dtype=np.uint8)
    cdef np.float32_t [:,:,::1] Idata_c = np.ascontiguousarray(Idata)
    cdef np.complex64_t [:,:,::1] obj_c = np.ascontiguousarray(obj)
    cdef np.complex64_t [:,:,::1] probe_c = np.ascontiguousarray(probe)
    # can't cast to continous array here, as it might copy and we update in-place
    cdef np.complex64_t [:,:,::1] exit_wave_c = exit_wave
    cdef np.int32_t [:,:,::1] addr_info_c = np.ascontiguousarray(addr_info)
    cdef np.complex64_t [:,::1] prefilter_c = None
    if not prefilter is None:
        prefilter_c = np.ascontiguousarray(prefilter)
    cdef np.complex64_t [:,::1] postfilter_c = None
    if not postfilter is None:
        postfilter_c = np.ascontiguousarray(postfilter)
    cdef float pbound_c = 0.0
    if not pbound is None:
        pbound_c = pbound
    errors = np.empty((3, mask.shape[0]), dtype=np.float32)
    cdef np.float32_t [:,::1] errors_c = errors
    cdef float alpha_c = alpha 
    cdef int doLLError = 1 if LL_error else 0
    cdef int doRealspaceError = 1 if do_realspace_error else 0
    cdef int doPbound = 0 if pbound == None else 1
    cdef int M = exit_wave.shape[0]
    cdef int A = exit_wave.shape[1]
    cdef int B = exit_wave.shape[2]
    cdef int ob_modes = obj.shape[0]
    cdef int C = obj.shape[1]
    cdef int D = obj.shape[2]
    cdef int pr_modes = probe.shape[0]
    cdef int N = mask.shape[0]
    # assertions to make sure dimensions are as expected
    assert(probe.shape[1] == A)
    assert(probe.shape[2] == B)
    assert(mask.shape[1] == A)
    assert(mask.shape[2] == B)
    assert(Idata.shape[0] == N)
    assert(Idata.shape[1] == A)
    assert(Idata.shape[2] == B)
    assert(addr_info.shape[0] == M)
    assert(errors.shape[1] == N)
    
    difference_map_fourier_constraint_c(
        <const unsigned char*>&mask_c[0],
        <const float*>&Idata_c[0,0,0],
        <const float*>&obj_c[0,0,0],
        <const float*>&probe_c[0,0,0],
        <float*>&exit_wave_c[0,0,0],
        <const int*>&addr_info_c[0,0,0],
        <const float*>&prefilter_c[0,0],
        <const float*>&postfilter_c[0,0],
        pbound_c, alpha_c, 
        doLLError,
        doRealspaceError,
        doPbound,
        M,
        N,
        A, B, C, D,
        ob_modes, pr_modes,
        <float*>&errors_c[0,0]
    )
    return errors

@cython.boundscheck(False)
@cython.wraparound(False)
def norm2f(input):
    cdef np.float32_t [::1] input_c = np.frombuffer(np.ascontiguousarray(input), dtype=np.float32)
    cdef int size = np.prod(input.shape)
    cdef float out = 0.0
    norm2_c(
        <const float*>&input_c[0],
        <float*>&out,
        size,
        0
    )
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def norm2c(input):
    cdef np.complex64_t [::1] input_c = np.frombuffer(np.ascontiguousarray(input), dtype=np.complex64)
    cdef int size = np.prod(input.shape)
    cdef float out = 0.0
    norm2_c(
        <const float*>&input_c[0],
        <float*>&out,
        size,
        1
    )
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def norm2(input):
    if input.dtype == np.float32:
        return norm2f(input)
    elif input.dtype == np.complex64:
        return norm2c(input)
    else:
        raise NotImplementedError("Norm2 is only implemented for single precision")

@cython.boundscheck(False)
@cython.wraparound(False)
def mass_center(A):
    if A.dtype != np.float32:
        raise NotImplementedError("Only single precision is supported")

    cdef np.float32_t [::1] input_c = np.frombuffer(np.ascontiguousarray(A), dtype=np.float32)
    cdef int i = A.shape[0]
    cdef int m = A.shape[1]
    cdef int n = 1
    if len(A.shape) == 3:
        n = A.shape[2]
    out = np.empty(len(A.shape), np.float32)
    cdef np.float32_t [::1] out_c = out
    mass_center_c(
        <const float*>&input_c[0],
        i, m, n, 
        <float*>&out_c[0]
    )
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def clip_complex_magnitudes_to_range(complex_input, clip_min, clip_max):
    if complex_input.dtype != np.complex64:
        raise NotImplementedError("Only single precision complex data supported")
    if not complex_input.flags['C_CONTIGUOUS']:
        raise NotImplementedError("Can only handle contiguous arrays, due to in-place updates")
    cdef int n = np.prod(complex_input.shape)
    cdef float c_min = clip_min
    cdef float c_max = clip_max
    # discard all dimensionality information
    cdef np.complex64_t [::1] input_c = np.frombuffer(np.ascontiguousarray(complex_input), dtype=np.complex64)
    clip_complex_magnitudes_to_range_c(
        <float*>&input_c[0],
        n, c_min, c_max
    )

@cython.boundscheck(False)
@cython.wraparound(False)
def extract_array_from_exit_wave(exit_wave, exit_addr, array_to_be_extracted, extract_addr, array_to_be_updated, update_addr, cfact, weights):
    cdef np.complex64_t [:,:,::1] exit_wave_c = np.ascontiguousarray(exit_wave)
    cdef np.int32_t [:,::1] exit_addr_c = np.ascontiguousarray(exit_addr).astype(np.int32)
    cdef np.complex64_t [:,:,::1] array_to_be_extracted_c = np.ascontiguousarray(array_to_be_extracted)
    cdef np.int32_t [:,::1] extract_addr_c = np.ascontiguousarray(extract_addr).astype(np.int32)
    cdef np.complex64_t [:,:,::1] array_to_be_updated_c = np.ascontiguousarray(array_to_be_updated)
    cdef np.int32_t [:,::1] update_addr_c = np.ascontiguousarray(update_addr).astype(np.int32)
    cdef np.complex64_t [:,:,::1] cfact_c = np.ascontiguousarray(cfact)
    cdef np.float32_t [::1] weights_c = np.ascontiguousarray(weights)
    cdef int A = exit_wave.shape[0]
    cdef int B = exit_wave.shape[1]
    cdef int C = exit_wave.shape[2]
    cdef int D = array_to_be_extracted.shape[0]
    cdef int E = array_to_be_extracted.shape[1]
    cdef int F = array_to_be_extracted.shape[2]
    cdef int G = array_to_be_updated.shape[0]
    cdef int H = array_to_be_updated.shape[1]
    cdef int I = array_to_be_updated.shape[2]
    extract_array_from_exit_wave_c(
        <const float*>&exit_wave_c[0,0,0],
        A, B, C,
        <const int*>&exit_addr_c[0,0],
        <const float*>&array_to_be_extracted_c[0,0,0],
        D, E, F,
        <const int*>&extract_addr_c[0,0],
        <float*>&array_to_be_updated_c[0,0,0],
        G, H, I,
        <const int*>&update_addr_c[0,0],
        <const float*>&weights_c[0],
        <const float*>&cfact_c[0,0,0]
    )

@cython.boundscheck(False)
@cython.wraparound(False)
def interpolated_shift(c, shift, do_linear=False):
    if not do_linear and all([int(s) != s for s in shift]):
        raise NotImplementedError("Bicubic interpolated shifts are not implemented yet")
    if c.dtype != np.complex64:
        raise NotImplementedError("Only complex single precision type supported")
    cdef int items = 0
    cdef int rows = 0
    cdef int columns = 0
    if len(c.shape) == 3:
        items = c.shape[0]
        rows = c.shape[1]
        columns = c.shape[2]
    else:
        items = 1
        rows = c.shape[0]
        columns = c.shape[1]
    cdef np.complex64_t [::1] c_c = \
        np.frombuffer(np.ascontiguousarray(c), dtype=np.complex64)
    out = np.zeros(c.shape, dtype=np.complex64)
    cdef np.complex64_t [::1] out_c = np.frombuffer(out, dtype=np.complex64)
    cdef float offsetRow = shift[0]
    cdef float offsetCol = shift[1]
    interpolated_shift_c(
        <const float*>&c_c[0],
        <float*>&out_c[0],
        items, rows, columns, 
        offsetRow, offsetCol,
        1
    )
    return out

def get_num_gpus():
    return get_num_gpus_c()

def get_gpu_compute_capability(dev):
    cdef int dev_c = dev
    return get_gpu_compute_capability_c(dev_c)

def select_gpu_device(dev):
    cdef int dev_c = dev
    select_gpu_device_c(dev_c)

def get_gpu_memory_mb(dev):
    cdef int dev_c = dev
    return get_gpu_memory_mb_c(dev_c)

def get_gpu_name(dev):
    cdef int dev_c = dev
    cdef string name = get_gpu_name_c(dev_c)
    return name.decode('UTF-8')

def reset_function_cache():
    reset_function_cache_c()

@cython.boundscheck(False)
@cython.wraparound(False)
def center_probe(probe, center_tolerance):
    # can't convert to contiguous array here, as it's in-place returned
    # we let Cython flag this if not contiguous
    cdef np.complex64_t [:,:,::1] probe_c = probe
    cdef float tol = center_tolerance
    cdef int i = probe.shape[0]
    cdef int m = probe.shape[1]
    cdef int n = probe.shape[2]
    center_probe_c(
        <float*>&probe_c[0,0,0],
        tol, i, m, n
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def difference_map_update_probe(obj, probe_weights, probe, exit_wave, addr_info, cfact_probe, probe_support=None):
    cdef np.complex64_t [:,:,::1] obj_c = np.ascontiguousarray(obj)
    cdef np.float32_t [::1] probe_weights_c = np.ascontiguousarray(probe_weights)
    # updated in-place, so can't use ascontiguousarray
    cdef np.complex64_t [:,:,::1] probe_c = probe
    cdef np.complex64_t [:,:,::1] exit_wave_c = np.ascontiguousarray(exit_wave)
    if not isinstance(addr_info, np.ndarray):
        addr_info = np.array(addr_info, dtype=np.int32)
    else:
        addr_info = addr_info.astype(np.int32)
    cdef np.int32_t [:,:,::1] addr_info_c = np.ascontiguousarray(addr_info)
    cdef np.complex64_t [:,:,::1] cfact_probe_c = np.ascontiguousarray(cfact_probe)
    cdef np.complex64_t [:,:,::1] probe_support_c = None
    if probe_support is not None:
        probe_support_c = np.ascontiguousarray(probe_support)
    cdef int A = exit_wave.shape[0]
    cdef int B = exit_wave.shape[1]
    cdef int C = exit_wave.shape[2]
    cdef int D = obj.shape[0]
    cdef int E = obj.shape[1]
    cdef int F = obj.shape[2]
    cdef int G = cfact_probe.shape[0]
    cdef int H = cfact_probe.shape[1]
    cdef int I = cfact_probe.shape[2]
    return difference_map_update_probe_c(
        <const float*>&obj_c[0,0,0],
        <const float*>&probe_weights_c[0],
        <float*>&probe_c[0,0,0],
        <const float*>&exit_wave_c[0,0,0],
        <const int*>&addr_info_c[0,0,0],
        <const float*>&cfact_probe_c[0,0,0],
        <const float*>&probe_support_c[0,0,0],
        A,B,C,D,E,F,G,H,I
    )

@cython.boundscheck(False)
@cython.wraparound(False)
def difference_map_update_object(obj, object_weights, probe, exit_wave, addr_info, cfact_object, ob_smooth_std=None, clip_object=None):
    # updated in-place, so can't use ascontiguousarray
    cdef np.complex64_t [:,:,::1] obj_c = obj
    cdef np.complex64_t [:,:,::1] probe_c = np.ascontiguousarray(probe)
    cdef np.float32_t [::1] object_weights_c = np.ascontiguousarray(object_weights)
    cdef np.complex64_t [:,:,::1] exit_wave_c = np.ascontiguousarray(exit_wave)
    if not isinstance(addr_info, np.ndarray):
        addr_info = np.array(addr_info, dtype=np.int32)
    else:
        addr_info = addr_info.astype(np.int32)
    cdef np.int32_t [:,:,::1] addr_info_c = np.ascontiguousarray(addr_info)
    cdef np.complex64_t [:,:,::1] cfact_object_c = np.ascontiguousarray(cfact_object)
    cdef float ob_smooth_std_c = 0
    cdef int doSmoothing = 0
    cdef int doClipping = 0
    if ob_smooth_std is not None:
        ob_smooth_std_c = ob_smooth_std
        doSmoothing = 1
    cdef float clip_min_c = 0
    cdef float clip_max_c = 0
    if clip_object is not None:
        clip_min_c = clip_object[0]
        clip_max_c = clip_object[1]
        doClipping = 1
    cdef int A = exit_wave.shape[0]
    cdef int B = exit_wave.shape[1]
    cdef int C = exit_wave.shape[2]
    cdef int D = probe.shape[0]
    cdef int E = probe.shape[1]
    cdef int F = probe.shape[2]
    cdef int G = obj.shape[0]
    cdef int H = obj.shape[1]
    cdef int I = obj.shape[2]

    difference_map_update_object_c(
        <float*>&obj_c[0,0,0],
        <const float*>&object_weights_c[0],
        <const float*>&probe_c[0,0,0],
        <const float*>&exit_wave_c[0,0,0],
        <const int*>&addr_info_c[0,0,0],
        <const float*>&cfact_object_c[0,0,0],
        ob_smooth_std_c, clip_min_c, clip_max_c,
        doSmoothing, doClipping,
        A,B,C,D,E,F,G,H,I
    )

@cython.boundscheck(False)
@cython.wraparound(False)
def difference_map_overlap_update(addr_info, cfact_object, cfact_probe, do_update_probe, exit_wave, ob, object_weights,
                                  probe, probe_support, probe_weights,max_iterations, update_object_first,
                                  obj_smooth_std, overlap_converge_factor, probe_center_tol, clip_object=None):
    # updated in-place, so can't use ascontiguousarray
    cdef np.complex64_t [:,:,::1] obj_c = ob
    cdef np.complex64_t [:,:,::1] probe_c = probe
    cdef np.float32_t [::1] object_weights_c = np.ascontiguousarray(object_weights)
    cdef np.float32_t [::1] probe_weights_c = np.ascontiguousarray(probe_weights)
    cdef np.complex64_t [:,:,::1] exit_wave_c = np.ascontiguousarray(exit_wave)
    cdef np.complex64_t [:,:,::1] cfact_object_c = np.ascontiguousarray(cfact_object)
    cdef np.complex64_t [:,:,::1] cfact_probe_c = np.ascontiguousarray(cfact_probe)
    cdef np.complex64_t [:,:,::1] probe_support_c = None
    if probe_support is not None:
        probe_support_c = np.ascontiguousarray(probe_support)
    if not isinstance(addr_info, np.ndarray):
        addr_info = np.array(addr_info, dtype=np.int32)
    else:
        addr_info = addr_info.astype(np.int32)
    cdef np.int32_t [:,:,::1] addr_info_c = np.ascontiguousarray(addr_info)
    cdef float overlap_converge_factor_c = overlap_converge_factor
    cdef int max_iterations_c = max_iterations

    cdef int doUpdateObjectFirst = 0
    cdef int doUpdateProbe = 0
    cdef int doSmoothing = 0
    cdef int doClipping = 0
    cdef int doCentering = 0

    cdef float ob_smooth_std_c = 0
    cdef float clip_min_c = 0
    cdef float clip_max_c = 0
    cdef float probe_center_tol_c = 0

    if update_object_first:
        doUpdateObjectFirst = 1
    if do_update_probe:
        doUpdateProbe = 1
    
    if obj_smooth_std is not None:
        ob_smooth_std_c = obj_smooth_std
        doSmoothing = 1
    
    if clip_object is not None:
        clip_min_c = clip_object[0]
        clip_max_c = clip_object[1]
        doClipping = 1
    
    if probe_center_tol is not None:
        probe_center_tol_c = probe_center_tol
        doCentering = 1
    
    cdef int A = exit_wave.shape[0]
    cdef int B = exit_wave.shape[1]
    cdef int C = exit_wave.shape[2]
    cdef int D = probe.shape[0]
    cdef int E = probe.shape[1]
    cdef int F = probe.shape[2]
    cdef int G = ob.shape[0]
    cdef int H = ob.shape[1]
    cdef int I = ob.shape[2]

    difference_map_overlap_constraint_c(
        <const int*>&addr_info_c[0,0,0],
        <const float*>&cfact_object_c[0,0,0],
        <const float*>&cfact_probe_c[0,0,0],
        <const float*>&exit_wave_c[0,0,0],
        <float*>&obj_c[0,0,0],
        <const float*>&object_weights_c[0],
        <float*>&probe_c[0,0,0],
        <const float*>&probe_support_c[0,0,0],
        <const float*>&probe_weights_c[0],
        ob_smooth_std_c,
        clip_min_c, clip_max_c,
        probe_center_tol_c, overlap_converge_factor_c,
        max_iterations_c, 
        doUpdateObjectFirst, doUpdateProbe, doSmoothing,
        doClipping, doCentering,
        A, B, C, D, E, F, G, H, I
    )

@cython.boundscheck(False)
@cython.wraparound(False)
def complex_gaussian_filter(input, mfs):
    cdef int ndims = len(input.shape)
    shape = np.ascontiguousarray(np.array(input.shape, dtype=np.int32))
    cdef np.int32_t [::1] shape_c = shape
    mfs = np.ascontiguousarray(np.array(mfs, dtype=np.float32))
    cdef np.float32_t [::1] mfs_c = mfs 
    if input.dtype != np.complex64:
        raise NotImplementedError("only complex64 type is supported")
    cdef np.complex64_t [::1] input_c = np.frombuffer(np.ascontiguousarray(input), dtype=np.complex64)
    out = np.empty_like(input)
    cdef np.complex64_t [::1] output_c = np.frombuffer(out, dtype=np.complex64)
    complex_gaussian_filter_c(
        <const float*>&input_c[0],
        <float*>&output_c[0],
        <const float*>&mfs_c[0],
        len(shape),
        <const int*>&shape_c[0]
    )
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def difference_map_iterator(diffraction, obj, object_weights, cfact_object, mask, probe, cfact_probe, probe_support,
                            probe_weights, exit_wave, addr, pre_fft, post_fft, pbound, overlap_max_iterations, update_object_first,
                            obj_smooth_std, overlap_converge_factor, probe_center_tol, probe_update_start, alpha=1,
                            clip_object=None, LL_error=False, num_iterations=1, do_realspace_error=True):
    cdef np.float32_t [:,:,::1] diffraction_c = np.ascontiguousarray(diffraction)
    # updated in-place, so can't use ascontiguousarray
    cdef np.complex64_t [:,:,::1] obj_c = obj
    cdef np.float32_t [::1] object_weights_c = np.ascontiguousarray(object_weights)
    cdef np.complex64_t [:,:,::1] cfact_object_c = np.ascontiguousarray(cfact_object)
    # converted to np.bool if needed!
    cdef np.uint8_t [::1] mask_c = np.frombuffer(np.ascontiguousarray(mask.astype(np.bool)), dtype=np.uint8)
    # updated in-place, so can't use ascontiguousarray
    cdef np.complex64_t [:,:,::1] probe_c = probe
    cdef np.complex64_t [:,:,::1] cfact_probe_c = np.ascontiguousarray(cfact_probe)
    cdef np.complex64_t [:,:,::1] probe_support_c = None
    if probe_support is not None:
        probe_support_c = np.ascontiguousarray(probe_support)
    cdef np.float32_t [::1] probe_weights_c = np.ascontiguousarray(probe_weights)
    # updated in-place, so can't use ascontiguousarray
    cdef np.complex64_t [:,:,::1] exit_wave_c = exit_wave
        
    if not isinstance(addr, np.ndarray):
        addr_info = np.array(addr, dtype=np.int32)
    else:
        addr_info = addr.astype(np.int32)
    cdef np.int32_t [:,:,::1] addr_info_c = np.ascontiguousarray(addr_info)
    cdef np.complex64_t [:,::1] pre_fft_c = np.ascontiguousarray(pre_fft)
    cdef np.complex64_t [:,::1] post_fft_c = np.ascontiguousarray(post_fft)
    errors = np.empty((num_iterations, 3, diffraction.shape[0]), dtype=np.float32)
    cdef np.float32_t [:,:,::1] errors_c = errors

    cdef float pbound_c = 0.0
    cdef int doPbound = 0
    if not pbound is None:
        pbound_c = pbound
        doPbound = 1

    cdef int overlap_max_iterations_c = overlap_max_iterations

    cdef int doUpdateObjectFirst = 0
    if update_object_first:
        doUpdateObjectFirst = 1
    
    cdef float ob_smooth_std_c = 0
    cdef int doSmoothing = 0
    if obj_smooth_std is not None:
        ob_smooth_std_c = obj_smooth_std
        doSmoothing = 1

    cdef float overlap_converge_factor_c = overlap_converge_factor

    cdef int doCentering = 0
    cdef float probe_center_tol_c = 0
    if probe_center_tol is not None:
        probe_center_tol_c = probe_center_tol
        doCentering = 1

    cdef int probe_update_start_c = probe_update_start
    cdef float alpha_c = alpha

    cdef int doClipping = 0
    cdef float clip_min_c = 0
    cdef float clip_max_c = 0
    if clip_object is not None:
        clip_min_c = clip_object[0]
        clip_max_c = clip_object[1]
        doClipping = 1

    cdef int do_LL_Error = 0
    if LL_error:
        do_LL_Error = 1
    
    cdef int doRealspaceError = 0
    if do_realspace_error:
        doRealspaceError = 1
    
    cdef int num_iterations_c = num_iterations
    
    cdef int A = exit_wave.shape[0]
    cdef int B = exit_wave.shape[1]
    cdef int C = exit_wave.shape[2]
    cdef int D = probe.shape[0]
    cdef int E = probe.shape[1]
    cdef int F = probe.shape[2]
    cdef int G = obj.shape[0]
    cdef int H = obj.shape[1]
    cdef int I = obj.shape[2]
    cdef int N = diffraction.shape[0]

    difference_map_iterator_c(
        <const float*>&diffraction_c[0,0,0],
        <float*>&obj_c[0,0,0],
        <const float*>&object_weights_c[0],
        <const float*>&cfact_object_c[0,0,0],
        <const unsigned char*>&mask_c[0],
        <float*>&probe_c[0,0,0],
        <const float*>&cfact_probe_c[0,0,0],
        <const float*>&probe_support_c[0,0,0],
        <const float*>&probe_weights_c[0],
        <float*>&exit_wave_c[0,0,0],
        <const int*>&addr_info_c[0,0,0],
        <const float*>&pre_fft_c[0,0],
        <const float*>&post_fft_c[0,0],
        <float*>&errors_c[0,0,0],
        pbound_c,
        overlap_max_iterations_c,
        doUpdateObjectFirst,
        ob_smooth_std_c,
        overlap_converge_factor_c,
        probe_center_tol_c,
        probe_update_start_c,
        alpha_c,
        clip_min_c, clip_max_c,
        do_LL_Error, doRealspaceError,
        num_iterations_c,
        A, B, C, D, E, F, G, H, I, N,
        doSmoothing, doClipping, doCentering, doPbound
    )
    return errors
