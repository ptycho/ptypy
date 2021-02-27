#from pytpy.array_based import COMPLEX_TYPE, FLOAT_TYPE
import numpy as np

from archive.cuda_extension.accelerate.cuda import difference_map_fourier_constraint
#from ptypy.accelerate.array_based.constraints import difference_map_fourier_constraint

from archive.cuda_extension.accelerate.cuda import difference_map_overlap_update
#from ptypy.accelerate.array_based.constraints import difference_map_overlap_update


from archive.cuda_extension.accelerate.cuda import far_field_error, realspace_error
#from ptypy.accelerate.array_based.error_metrics import far_field_error, realspace_error

#from ptypy.accelerate.array_based.error_metrics import log_likelihood
from archive.cuda_extension.accelerate.cuda import log_likelihood

from archive.cuda_extension.accelerate.cuda import difference_map_realspace_constraint
#from ptypy.accelerate.array_based.object_probe_interaction import difference_map_realspace_constraint

from archive.cuda_extension.accelerate.cuda import scan_and_multiply
#from ptypy.accelerate.array_based.object_probe_interaction import scan_and_multiply

from archive.cuda_extension.accelerate.cuda import renormalise_fourier_magnitudes
#from ptypy.accelerate.array_based.constraints import renormalise_fourier_magnitudes

from archive.cuda_extension.accelerate.cuda import get_difference
#from ptypy.accelerate.array_based.constraints import get_difference

from archive.cuda_extension.accelerate.cuda import abs2
#from ptypy.accelerate.array_based.array_utils import abs2

from archive.cuda_extension.accelerate.cuda import sum_to_buffer
#from ptypy.accelerate.array_based.array_utils import sum_to_buffer

from archive.cuda_extension.accelerate.cuda import farfield_propagator
#from ptypy.accelerate.array_based.propagation import farfield_propagator


FLOAT_TYPE = np.float32
COMPLEX_TYPE = np.complex64

# def difference_map_fourier_constraint(mask, Idata, obj, probe, exit_wave, addr_info, prefilter, postfilter, pbound=None, alpha=1.0, LL_error=True, do_realspace_error=True):
#     '''
#     This kernel just performs the fourier renormalisation.
#     :param mask. The nd mask array
#     :param diffraction. The nd diffraction data
#     :param farfield_stack. The current iterant.
#     :param addr. The addresses of the stacks.
#     :return: The updated iterant
#             : fourier errors
#     '''

#     probe_object = scan_and_multiply(probe, obj, exit_wave.shape, addr_info)

#     # Buffer for accumulated photons
#     # For log likelihood error # need to double check this adp
#     if LL_error is True:
#         print("probe_object={}, {}".format(probe_object.shape, probe_object.dtype))
#         print("mask={}, {}".format(mask.shape, mask.dtype))
#         print("Idata={}, {}".format(Idata.shape, Idata.dtype))
#         print("prefilter={}, {}".format(prefilter.shape, prefilter.dtype))
#         print("postfilter={}, {}".format(postfilter.shape, postfilter.dtype))
#         print("addr_info={}, {}".format(addr_info.shape, addr_info.dtype))
#         err_phot = log_likelihood(probe_object, mask, Idata, prefilter, postfilter, addr_info)
#         print("err_phot={}, {}".format(err_phot.shape, err_phot.dtype))
#     else:
#         err_phot = np.zeros(Idata.shape[0], dtype=FLOAT_TYPE)
    
    
#     constrained = difference_map_realspace_constraint(probe_object, exit_wave, alpha)
#     #print("constrained=\n{}".format(constrained[0]))
#     #constr = constrained.flatten()
#     #for i in xrange(1000):
#     #    if abs(constr[i]) > 0.0:
#     #        print("{}: {}".format(i, constr[i]))

#     f = farfield_propagator(constrained, prefilter, postfilter, direction='forward')
#     pa, oa, ea, da, ma = zip(*addr_info)
#     af2 = sum_to_buffer(abs2(f), Idata.shape, ea, da, dtype=FLOAT_TYPE)

#     fmag = np.sqrt(np.abs(Idata))
#     af = np.sqrt(af2)
#     # # Fourier magnitudes deviations(current_solution, pbound, measured_solution, mask, addr)
#     err_fmag = far_field_error(af, fmag, mask)

#     # print("f={}, {}".format(f.shape, f.dtype))
#     # print("af={}, {}".format(af.shape, af.dtype))
#     # print("fmag={}, {}".format(fmag.shape, fmag.dtype))
#     # print("mask={}, {}".format(mask.shape, mask.dtype))
#     # print("err_fmag={}, {}".format(err_fmag.shape, err_fmag.dtype))
#     # print("addr_info={}, {}".format(addr_info.shape, addr_info.dtype))
#     # print("pbound={}".format(pbound))
#     vectorised_rfm = renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound)

#     # print("vectorised_rfm={}, {}".format(vectorised_rfm.shape, vectorised_rfm.dtype))
    
#     backpropagated_solution = farfield_propagator(vectorised_rfm,
#                                                   postfilter.conj(),
#                                                   prefilter.conj(),
#                                                   direction='backward')

    
#     df = get_difference(addr_info, alpha, backpropagated_solution, err_fmag, exit_wave, pbound, probe_object)

    

#     exit_wave += df
#     if do_realspace_error:
#         ea_first_column = np.array(ea)[:, 0]
#         da_first_column = np.array(da)[:, 0]
#         err_exit = realspace_error(df, ea_first_column, da_first_column, Idata.shape[0])
#     else:
#         err_exit = np.zeros((Idata.shape[0]))

#     if pbound is not None:
#         err_fmag /= pbound

#     #print("err_fmag: {}".format(err_fmag.shape))
#     #print("err_phot: {}".format(err_phot.shape))
#     #print("err_exit: {}".format(err_exit.shape))
#     return np.array([err_fmag, err_phot, err_exit])



def difference_map_iterator(diffraction, obj, object_weights, cfact_object, mask, probe, cfact_probe, probe_support,
                            probe_weights, exit_wave, addr, pre_fft, post_fft, pbound, overlap_max_iterations, update_object_first,
                            obj_smooth_std, overlap_converge_factor, probe_center_tol, probe_update_start, alpha=1,
                            clip_object=None, LL_error=False, num_iterations=1):
    curiter = 0

    errors = np.zeros((num_iterations, 3, len(diffraction)), dtype=FLOAT_TYPE)
    for it in range(num_iterations):
        if (((it+1) % 10) == 0) and (it>0):
            print("iteration:%s" % (it+1)) # it's probably a good idea to print this if possible for some idea of progress
        # numpy dump here for 64x64 and 4096x4096

        #print("mask: {}".format(mask.shape))
        #print("diffraction: {}".format(diffraction.shape))
        #print("obj: {}".format(obj.shape))
        #print("probe: {}".format(probe.shape))
        #print("exit_wave: {}".format(exit_wave.shape))
        #print("addr: {}".format(addr.shape))
        #print("prefilter: {}".format(pre_fft.shape))
        #print("postfiler: {}".format(post_fft.shape))
        err = difference_map_fourier_constraint(mask,
                                                   diffraction,
                                                   obj,
                                                   probe,
                                                   exit_wave,
                                                   addr,
                                                   prefilter=pre_fft,
                                                   postfilter=post_fft,
                                                   pbound=pbound,
                                                   alpha=alpha,
                                                   LL_error=LL_error)
        #print("err: {}".format(err.shape))
        errors[it] = err

        do_update_probe = (probe_update_start <= curiter)
        difference_map_overlap_update(addr,
                                      cfact_object,
                                      cfact_probe,
                                      do_update_probe,
                                      exit_wave,
                                      obj,
                                      object_weights,
                                      probe,
                                      probe_support,
                                      probe_weights,
                                      overlap_max_iterations,
                                      update_object_first,
                                      obj_smooth_std,
                                      overlap_converge_factor,
                                      probe_center_tol,
                                      clip_object=clip_object)
        curiter += 1
    return errors