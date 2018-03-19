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
def log_likelihood(probe_obj, mask, exit_wave, Idata, prefilter, postfilter, addr):
    cdef np.complex64_t [:,:,::1] probe_obj_c = np.ascontiguousarray(probe_obj)
    cdef np.uint8_t [::1] mask_c = np.frombuffer(np.ascontiguousarray(mask), dtype=np.uint8)
    cdef np.complex64_t [:,:,::1] exit_wave_c = np.ascontiguousarray(exit_wave)
    cdef np.float32_t [:,:,::1] Idata_c = np.ascontiguousarray(Idata)
    cdef np.complex64_t [:,::1] prefilter_c = np.ascontiguousarray(prefilter)
    cdef np.complex64_t [:,::1] postfilter_c = np.ascontiguousarray(postfilter)
    view_dlayer=0
    addr_info = addr[:, view_dlayer]
    cdef np.int32_t [:,:,::1] addr_info_c = np.ascontiguousarray(addr_info)
    out = np.empty([92], np.float32)
    cdef np.float32_t [::1] out_c = out
    cdef int i = probe_obj.shape[0]
    cdef int m = probe_obj.shape[1]
    cdef int n = probe_obj.shape[2] 
    log_likelihood_c(
        <const float*>&probe_obj_c[0,0,0],
        <const unsigned char*>&mask_c[0],
        <const float*>&exit_wave_c[0,0,0],
        <const float*>&Idata_c[0,0,0],
        <const float*>&prefilter_c[0,0],
        <const float*>&postfilter_c[0,0],
        <const int*>&addr_info_c[0,0,0],
        <float*>&out_c[0],
        i, m, n, addr_info.shape[0]
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
    cdef int n = np.product(cin.shape)

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
    cdef np.uint8_t [::1] mask_c = np.frombuffer(np.ascontiguousarray(mask), dtype=np.uint8)
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
def realspace_error(difference):
    cdef np.complex64_t [:,:,::1] difference_c = np.ascontiguousarray(difference)
    out = np.empty((difference.shape[0],), dtype=np.float32)
    cdef np.float32_t [::1] out_c = out
    cdef i = difference.shape[0]
    cdef m = difference.shape[1]
    cdef n = difference.shape[2]
    realspace_error_c(
        <const float*>&difference_c[0,0,0],
        <float*>&out_c[0],
        i, m, n
    )
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def get_difference(addr_info, alpha, backpropagated_solution, err_fmag, exit_wave, pbound, probe_object):
    #addr_info: (92, 5, 3), int32
    #alpha: float64
    #backpropagated_solution: (92, 256, 256), complex128
    #err_fmag: (92,), float64
    #exit_wave: (92, 256, 256), complex64
    #pbound: float64
    #probe_object: (92, 256, 256), complex64
    #out: (92, 256, 256), complex128
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
    cdef np.uint8_t [::1] mask_c = np.frombuffer(np.ascontiguousarray(mask), dtype=np.uint8)
    cdef np.float32_t [::1] err_fmag_c = np.ascontiguousarray(err_fmag.astype(np.float32))
    cdef np.int32_t [:,:,::1] addr_info_c = np.ascontiguousarray(addr_info)
    cdef pbound_c = 0.0
    if not pbound is None:
        pbound_c = pbound
    out = np.empty_like(f)
    cdef np.complex64_t [:,:,::1] out_c = out
    cdef int i = f.shape[0]
    cdef int m = f.shape[1]
    cdef int n = f.shape[2]
    renormalise_fourier_magnitudes_c(
        <const float*>&f_c[0,0,0],
        <const float*>&af_c[0,0,0],
        <const float*>&fmag_c[0,0,0],
        <const unsigned char*>&mask_c[0],
        <const float*>&err_fmag_c[0],
        <const int*>&addr_info_c[0,0,0],
        pbound_c,
        <float*>&out_c[0,0,0],
        i, m, n,
        0 if pbound == None else 1
    )
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def difference_map_fourier_constraint(mask, Idata, obj, probe, exit_wave, addr, prefilter, postfilter, pbound=None, alpha=1.0, LL_error=True, do_realspace_error=True):
    cdef np.uint8_t [::1] mask_c = np.frombuffer(np.ascontiguousarray(mask), dtype=np.uint8)
    cdef np.float32_t [:,:,::1] Idata_c = np.ascontiguousarray(Idata)
    cdef np.complex64_t [:,:,::1] obj_c = np.ascontiguousarray(obj)
    cdef np.complex64_t [:,:,::1] probe_c = np.ascontiguousarray(probe)
    # can't cast to continous array here, as it might copy and we update in-place
    cdef np.complex64_t [:,:,::1] exit_wave_c = exit_wave
    view_dlayer = 0 # what is this?
    addr_info = addr[:,(view_dlayer)] # addresses, object references
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
    errors = np.empty((3, exit_wave.shape[0]), dtype=np.float32)
    cdef np.float32_t [:,::1] errors_c = errors
    cdef float alpha_c = alpha 
    cdef int doLLError = 1 if LL_error else 0
    cdef int doRealspaceError = 1 if do_realspace_error else 0
    cdef int doPbound = 0 if pbound == None else 1
    cdef int ews0 = exit_wave.shape[0]
    cdef int ews1 = exit_wave.shape[1]
    cdef int ews2 = exit_wave.shape[2]
    cdef int os1 = obj.shape[1]
    cdef int os2 = obj.shape[2]
    cdef int ps1 = probe.shape[1]
    cdef int ps2 = probe.shape[2]
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
        ews0,
        ews1,
        ews2,
        os1, os2,
        ps1, ps2,
        <float*>&errors_c[0,0]
    )
    return errors