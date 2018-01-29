'''
All propagation based kernels 
'''
import numpy as np
import scipy as sci
from object_probe_interaction import difference_map_realspace_constraint, scan_and_multiply

def farfield_propagator(data_to_be_transformed, prefilter=None, postfilter=None, direction='forward'):
    '''
    performs a fourier transform on the nd exit wave stack. FFT shift and normalisation performed by 
    multiplication with prefilter and postfilter 
    :param data_to_be_transformed. The nd stack of the current iterant.
    :param prefilter. The filter to multiply before fourier transforming. Default: None.
    :param postfilter. The filter to multiply after fourier transforming. Default: None.
    :param direction. The direction of the transform forward or backward. Default: Forward.
    :return: The transformed stack.
    '''

    if direction is 'forward':
        # fft = np.fft.fft2
        fft = sci.fftpack.fft2
        sc = 1.0 / np.sqrt(np.prod(data_to_be_transformed.shape[-2:]))

    elif direction is 'backward':
        # fft  = np.fft.ifft2
        fft = sci.fftpack.ifft2
        sc = np.sqrt(np.prod(data_to_be_transformed.shape[-2:]))
    
    if (prefilter is None) and (postfilter is None):
        return fft(data_to_be_transformed, axes=(-2,-1)) * sc
    elif (prefilter is None) and (postfilter is not None):
        return np.multiply(postfilter, fft(data_to_be_transformed, axes=(-2,-1))) * sc
    elif (prefilter is not None) and (postfilter is None):
        return fft(np.multiply(data_to_be_transformed, prefilter), axes=(-2,-1))* sc
    elif (prefilter is not None) and (postfilter is not None):
        return np.multiply(postfilter, fft(np.multiply(data_to_be_transformed, prefilter), axes=(-2,-1))) * sc

def sqrt_abs(diffraction):
    return np.sqrt(np.abs(diffraction))





def calculate_err_phot(Idata, LL, addr_info, mask):
    err_phot = np.zeros(Idata.shape[0])
    for _pa, _oa, _ea, da, ma in addr_info:
        # LLminI2 = np.power((LL[da[0]] - Idata[da[0]]), 2)
        # timessum = np.sum(np.divide(np.multiply(mask[ma[0]], LLminI2), np.add(Idata[da[0]], 1.0)))
        # err_phot[da[0]] = np.divide(timessum, np.prod(LL[da[0]].shape))
        err_phot[da] = (np.sum(mask[ma] * (LL[da] - Idata[da])**2 / (Idata[da] + 1.))
                    / np.prod(LL[da].shape))
    return err_phot


def difference_map_fourier_constraint(mask, Idata, obj, probe, exit_wave, addr, prefilter, postfilter, pbound=None, alpha=1.0, LL_error=True):
    '''
    This kernel just performs the fourier renormalisation.
    :param mask. The nd mask array
    :param diffraction. The nd diffraction data
    :param farfield_stack. The current iterant.
    :param addr. The addresses of the stacks.
    :return: The updated iterant
            : fourier errors
    '''
    # Buffer for accumulated photons
    view_dlayer=0
    addr_info = addr[:, view_dlayer]
    af2 = np.zeros_like(Idata)
    sh = Idata.shape
    # For log likelihood error # need to double check this adp
    if LL_error is True:
        LL = np.zeros_like(Idata, dtype=Idata.dtype)
        po = scan_and_multiply(probe, obj, sh, addr_info)
        fing = farfield_propagator(po[0], prefilter, postfilter, direction='forward')
        pof = (fing * fing.conj()).real
        import pylab as plt
        plt.ion()
        plt.figure(1)
        plt.imshow(np.abs(po[10]))
        plt.colorbar()
        plt.show()
        plt.savefig('/tmp/gpu_po.png')
        for _pa, _oa, ea, da, _ma in addr_info:
            # print da[0], ea[0]
            LL[da[0]] += pof[ea[0]]

        err_phot = calculate_err_phot(Idata, LL, addr_info, mask)
    else:
        err_phot = np.zeros(sh[0])
    return err_phot
    # # Propagate the exit waves
    # constrained = difference_map_realspace_constraint(obj, probe, exit_wave, addr, alpha)
    # f = farfield_propagator(constrained, prefilter, postfilter, direction='forward')
    # for _pa, _oa, ea, da, _ma in addr_info:
    #     af2[da[0]] += np.power(np.abs(f[ea[0]]), 2)
    #
    # fmag = np.sqrt(np.abs(Idata))
    # af = np.sqrt(af2)
    #
    # # Fourier magnitudes deviations
    # fdev = np.subtract(af, fmag)
    # err_fmag = np.sum(np.multiply(mask, np.power(fdev, 2))) / np.sum(mask, axis=(0, 1))
    # err_exit = 0.
    #
    # probe_object = scan_and_multiply(probe, obj, sh, addr_info)
    # fm = None
    # for _pa, _oa, ea, da, ma in addr_info:
    #     if (pbound is None) or (err_fmag[da[0]] > pbound[da[0]]):
    #         # No power bound
    #         if pbound is None:
    #             fm = np.zeros_like(exit_wave)
    #             fm[ea] = (1 - mask[ma[0]]) + mask[ma[0]] * fmag[da[0]] / (af[da[0]] + 1e-10)
    #         elif err_fmag[da] > pbound[da]:
    #             # Power bound is applied
    #             fm = np.zeros_like(exit_wave)
    #             fm[ea] = (1 - mask[ma[0]]) + mask[ma[0]] * (fmag[da[0]] + fdev[da[0]] * np.sqrt(pbound[da[0]] / err_fmag[da[0]])) / (af[da[0]] + 1e-10)
    #
    #
    # if fm is None:
    #     df = np.multiply(alpha, (np.subtract(probe_object, exit_wave)))
    # else:
    #     backpropagated_solution = farfield_propagator(np.multiply(fm, f), prefilter, postfilter, direction='backward')
    #     df = np.subtract(backpropagated_solution, probe_object)
    #
    # exit_wave += df
    # err_exit += np.mean(np.power(np.abs(df), 2))
    #
    # if pbound is not None:
    #     # rescale the fmagnitude error to some meaning !!!
    #     # PT: I am not sure I agree with this.
    #     err_fmag /= pbound
    #
    # return exit_wave, np.array([err_fmag, err_phot, err_exit])



