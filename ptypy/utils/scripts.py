"""
Longer script-like functions.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
from . import parallel
import urllib.request, urllib.error, urllib.parse
from scipy import ndimage as ndi

from . import array_utils as au
from .misc import expect2
# from .. import io # FIXME: SC: when moved here, io fails

__all__ = ['hdr_image', 'diversify', 'cxro_iref', 'xradia_star', 'png2mpg',
           'mass_center', 'phase_from_dpc', 'radial_distribution',
           'stxm_analysis', 'stxm_init', 'load_from_ptyr', 'remove_hot_pixels']


def diversify(A, noise=None, shift=None, power=1.0):
    """
    Add diversity to 3d numpy array `A`, *acts in-place*.

    Parameters
    ----------
    A : np.array
        Input numpy array.

    noise : 2-tuple or 4-tuple
        For detailed description see :any:`ptypy.utils.parallel.MPInoise2d`

    power : float, tuple
        Relative power of layers with respect to the first (0) layer.
        Can be scalar or tuple / array

    shift : float, tuple
        Relative shift of layers with respect to the first (0) layer.
        Can be scalar or tuple / array
        **not implemented yet**

    See also
    --------
    ptypy.utils.parallel.MPInoise2d
    """
    # Nothing to do if only one mode
    if A.shape[0] == 1:
        return

    if noise is not None:
        noise = parallel.MPInoise2d(A.shape, *noise)
        # No noise where the main mode is
        noise[0] = 1.0
        A *= noise

    if shift is not None:
        raise NotImplementedError(
            'Diversity introduced by lateral shifts is not yet implemented.')

    # Expand power to length
    p = (power,) if np.isscalar(power) else tuple(power)
    # Check
    append = A.shape[0] - 1 - len(p)
    if append >= 1:
        p += (p[-1],) * append
    else:
        p = p[:A.shape[0] - 1]
    power = np.array((1.0,) + p).reshape(
        (A.shape[0],) + (1,) * (len(A.shape) - 1))
    power /= power.sum()
    A *= np.sqrt(power)

def hdr_image(img_list, exp_list, thresholds=[3000,50000], dark_list=[],
              avg_type='highest', mask_list=[], ClipLongestExposure=False,
              ClipShortestExposure=False):
    """
    Generate a high dynamic range image from a list of images `img_list`
    and corresponding exposure information in `exp_list`.

    Parameters
    ----------
    img_list : list
        Sequence of images (as 2d np.ndarray)

    exp_list : list of float
        Associated exposures to each element of above sequence `img_list`

    thresholds : list, 2-tuple
        Tuple of lower limit (noise floor) and upper limit (overexposure)
        in the images.

    dark_list : list
        Single frame or sequence of dark images (as 2d np.array) of the
        same length as `img_list`. These frames are used for dark-field
        correction

    avg_type : str
        Type of combining all valid pixels:

          - `'highest'`, the next longest exposure is used to replace
            overexposed pixels.
          - `<other_string>`, overexposed pixels are replaced by the
            pixel average of all other images with valid pixel values
            for that pixel.

    mask_list : list
        Single frame or sequence of 2d np.array.
        Provide additional masking (dead pixels, hot pixels)

    ClipLongestExposure : bool
        If True, also mask the noise floor in the longest exposure.

    ClipShortestExposure : bool
        if True, also mask the overexposed pixels in the shortest exposure.

    Returns
    -------
    out : ndarray
        Combined hdr-image with shape of one of the frame in `img_list`

    Examples
    --------
    >>> from ptypy import io
    >>> dark_list, meta = io.image_read('/path/to/dark/images/ccd*.raw')
    >>> img_list, meta= io.image_read('/path/to/images/ccd*.raw')
    >>> exp_list = [meta[j]['exposure_time__key'] for j in range(len(meta))]
    >>> hdr, masks = hdr_image(img_list, exp_list, dark_list=dark_list)


    """
    min_exp = min(exp_list)
    #print min_exp
    max_exp = max(exp_list)
    #print max_exp
    if len(mask_list) == 0:
        mask_list = [np.ones(img.shape) for img in img_list]
    elif len(mask_list) == 1:
        mask_list = mask_list * len(img_list)

    if len(dark_list) == 0:
        dark_list = [np.zeros(img.shape) for img in img_list]
    elif len(dark_list) == 1:
        dark_list = dark_list * len(img_list)
    # Convert to floats except for mask_list
    img_list = [img.astype(float) for img in img_list]
    dark_list = [dark.astype(float) for dark in dark_list]
    exp_list = [float(exp) for exp in exp_list]
    mask_list = [mask.astype(np.int) for mask in mask_list]

    for img, dark, exp,mask in zip(img_list, dark_list,exp_list,mask_list):
        img[:] = abs(img - dark)
        #img[img < 0] = 0
        #figure(); imshow(img); colorbar()
        maskhigh = ndi.binary_erosion(img < thresholds[1])
        masklow = ndi.binary_dilation(img > thresholds[0])
        if abs(exp - min_exp) / exp < 0.01 and not ClipShortestExposure:
            mask *= masklow
        elif abs(exp - max_exp) / exp < 0.01 and not ClipLongestExposure:
            mask *= maskhigh
        else:
            mask *= masklow*maskhigh

    if avg_type == 'highest':
        ix = list(np.argsort(exp_list))
        ix.reverse()
        mask_sum = mask_list[ix[0]]
        #print ix, img_list
        img_hdr = np.array(img_list[ix[0]]) * max_exp / exp_list[ix[0]]
        for j in range(1, len(ix)):
            themask = (1 - mask_sum) * mask_list[ix[j]]
            #figure(); imshow(mask_sum);
            #figure(); imshow(themask);
            mask_sum[themask.astype(bool)] = 1
            img_hdr[themask.astype(bool)] = (img_list[
                                                 ix[j]][themask.astype(bool)]
                                             * max_exp/exp_list[ix[j]])
    else:
        mask_sum = np.zeros_like(mask_list[0]).astype(np.int)
        img_hdr = np.zeros_like(img_list[0])
        for img, exp, mask in zip(img_list,exp_list,mask_list):
            img = img * max_exp/exp
            img_hdr += img * mask
            mask_sum += mask
        mask_sum[mask_sum == 0] = 1
        img_hdr = img_hdr / (mask_sum*1.)

    return img_hdr,mask_list

def png2mpg(listoffiles, framefile='frames.txt', fps=5, bitrate=2000,
            codec='wmv2', Encode=True, RemoveImages=False):
    r"""
    Makes a movie (\*.mpg) from a collection of \*.png or \*.jpeg frames.
    *Requires* binary of **mencoder** installed on system

    Parameters
    ----------
    listoffiles : list of str
        A list of paths to files. Each file will be reinterpreted as a
        collection files in the same directory, e.g. only the first file
        in a series needs to be selected. All series for which a first
        file was given will be concatenated in the movie. May also be
        a textfile with a listing of frames in which case the creation
        of an intermediate file of frame paths will be skipped.

    framefile : str
        Filepath, the respective file will be created to store a list
        of all frames. This file will be used by mencoder later.

    fps : scalar
        Frames per second in final movie

    bitrate : int
        Encoding detail, determines video quality

    codec : str
        Defines the codec to Use

    Encode : bool
        If True, video will be encoded calling mencoder
        If False, mencoder will not be called, but the necessary command
        expression will be returned instead. Very well suited for a dry-run

    RemoveImages : bool
        If True, all images referred to by framefile are deleted except
        for the last frame.

    Returns
    -------
    cmd : str
        Command string for the shell to encode the video later manually.

    Examples
    --------
    >>> from ptypy.utils.scripts import png2mpg
    >>> png2mpg(['/path/to/image_000.png'])

    The following happens:
    1) search for files similar to image_*.png in '/path/to/'
    2) found files get listed in a file '/path/to/frames.txt'
    3) calls mencoder to use that file to encode a movie with the default args.
    4) movie is in the same folder as 'frames.txt' and is called 'frames.mpg'


    >>> png2mpg(['/path1/to/imageA_040.png', '/path2/to/imageB_001.png'],
    >>>         framefile='./banana.txt')

    Generates list file 'banana_text' in current folder. The list file
    contains in order every path compatible with the wildcards
    '/path1/to/imageA_*.png' and '/path2/to/imageB_*.png'

    >>> str = png2mpg(['/path/to/image_000.png'], Encode=False)

    Returns encoder argument string. Use os.system(encoderstring) for
    encoding manually later

    """
    import os
    import glob
    import re
    framelist = []
    if str(listoffiles).endswith('.txt'):
        newframefile = open(listoffiles, 'r')
        framelist = [line.strip() for line in newframefile]
        newframefile.close()
        ff = listoffiles
    else:
        # In case of single file path.
        if str(listoffiles) == listoffiles:
            listoffiles = [listoffiles]

        for frame_or_list in listoffiles:
            if not os.path.isfile(frame_or_list):
                raise ValueError('File %s not found' % frame_or_list)
            else:
                head, tail = os.path.split(frame_or_list)
                # In case of executing the script in the same directory.
                if str(head) == '':
                    head = '.'

                if os.path.splitext(tail)[1] in ['.txt', '.dat']:
                    print('Found text file - try to interpret it as a list of '
                          'image files.')
                    temp = open(frame_or_list)
                    line = 'uiui'
                    while 'EOF' not in line and line != '':
                        line = temp.readline()
                        content = line.strip()
                        #print content
                        # Look if the referenced file exists and then add the
                        # file to the list.
                        if os.path.isfile(content):
                            framelist.append(content)
                        elif os.path.isfile(head + os.sep+content):
                            framelist.append(head + os.sep+content)
                        else:
                            print('File reference %s not found, continuing..'
                                  % content)
                            continue
                    temp.close()
                else:
                    # Trying to find similar images.
                    body, imagetype = os.path.splitext(tail)
                    # Replace possible numbers by a wildcard.
                    newbody = re.sub(r'\d+', '*', body)
                    wcard = head + os.sep + newbody + imagetype
                    #print wcard
                    imagfiles = glob.glob(wcard)
                    imagfiles.sort()
                    framelist += imagfiles

        if os.path.split(framefile)[0] == '':
            ff = head + os.sep + framefile
        else:
            ff = framefile

        newframefile = open(ff, 'w')
        #newframefile.writelines(framelist)
        for frame in framelist:
            newframefile.write('/' + os.path.relpath(frame, '/') + '\n')
        newframefile.close()

    last = os.path.relpath(framelist[-1])
    savelast = os.path.split(last)[0] + '/last' + os.path.splitext(last)[1]
    #print last
    #print savelast
    os.system('cp %s %s' % (last, savelast))

    frametype = os.path.splitext(last)[1].split('.')[-1]
    body = os.path.splitext(ff)[0]

    mencode_dict = dict(
        listf = os.path.relpath(ff),
        outputf = os.path.relpath(body),
        fps = fps,
        frametype = frametype,
        bitrate = bitrate,
        codec = codec
    )

    encodepattern = ''.join(['mencoder ',
                             'mf://@%(listf)s ',
                             '-o %(outputf)s.mpg ',
                             '-mf type=%(frametype)s:fps=%(fps)d ',
                             '-ovc lavc ',
                             '-lavcopts vbitrate=%(bitrate)d:vcodec=%(codec)s ',
                             '-oac copy'])

    encoderstring = encodepattern % mencode_dict

    if Encode:
        try:
            os.system(encoderstring)
        except OSError:
            print('Encoding failed - exiting')
        if RemoveImages:
            nff = open(ff, 'r')
            line = 'haha'
            while line != '':
                line = nff.readline().strip()
                print('Removing %s' % line)
                #os.remove(line)
                try:
                    os.remove(line)
                except OSError:
                    print(OSError)
                    print('Removing %s failed .. continuing' % line)
                    continue
            nff.close()
            if line == '':
                try:
                    print('Removing %s' % ff)
                    os.remove(ff)
                except OSError:
                    print('Removing %s failed' % ff)
    else:
        return encoderstring

def xradia_star(sh, spokes=48, std=0.5, minfeature=5, ringfact=2, rings=4,
                contrast=1., Fast=False):
    """
    Creates an Xradia-like star pattern on an array of shape `sh`.

    Works superb as test pattern in ptychography.

    *requires scipy*

    Parameters
    ----------
    sh : int, tuple
        Shape of requested array.

    std: float
        "Resolution" of xradia star, i.e. standard deviation of the
         error function used for smoothing the edges (in pixel).

    spokes : int, optional
        Number of spokes.

    minfeature : float, optional
        Spoke width at the smallest (inner) tip (in pixel).

    ringfact : float
        Increase in feature size from ring to ring (factor). Determines
        position of the rings.

    rings : int
        Number of rings with spokes.

    contrast : float
        Minimum contrast, set to zero for a gradual increase of the profile
        from zero to 1, set to 1 for no gradient in profile.

    Fast : bool
        If set to False, the error function is evaluated at the edges.
        Preferred choice when using fft, as edges are less prone to
        antialiasing in this case. If set to True, simple boolean
        comparison will be used instead to draw the edges and the
        result is later smoothed with a gaussian filter. Preferred choice
        for impatient users, as this choice is roughly a factor 2 faster
        for larger arrays

    Examples
    --------
    >>> from ptypy.utils.scripts import xradia_star
    >>> # Base configuration
    >>> X1 = xradia_star(1024)
    >>> # Few spokes single ring
    >>> X2 = xradia_star(1024, 12, std=4, rings=1, minfeature=10, ringfact=10)
    >>> # Very fine plus gradient
    >>> X3 = xradia_star(1024, 64, std = 0.2, rings = 10, minfeature=1,
    >>>                  contrast=0)
    >>> from matplotlib import pyplot as plt
    >>> ax = plt.subplot(131)
    >>> ax.imshow(X1, cmap='gray')
    >>> ax = plt.subplot(132)
    >>> ax.imshow(X2, cmap='gray')
    >>> ax = plt.subplot(133)
    >>> ax.imshow(X3, cmap='gray')
    >>> plt.show()

    """
    from scipy.ndimage import gaussian_filter
    from scipy.special import erf

    def step(x, a, std=0.5):
        if not Fast:
            return 0.5 * erf((x - a) / (std*2)) + 0.5
        else:
            return (x > a).astype(float)

    def rect(x,a):
        return step(x, -a / 2., std) * step(-x, -a / 2., std)

    def rectint(x, a, b):
        return step(x, a, std) * step(-x, -b, std)

    sh = expect2(sh)
    ind = np.indices(sh)
    cen = (sh - 1) / 2.0
    ind = ind-cen.reshape(cen.shape+len(cen)*(1,))
    z = ind[1] + 1j*ind[0]
    spokeint, spokestep = np.linspace(0.0 * np.pi,
                                      1.0 * np.pi,
                                      spokes // 2,
                                      False,
                                      True)
    spokeint += spokestep / 2

    r = np.abs(z)
    r0 = (minfeature / 2.0) / np.sin(spokestep / 4.)
    rlist = []
    rin = r0
    for ii in range(rings):
        if rin > max(sh) / np.sqrt(2.):
            break
        rin *= ringfact
        rlist.append((rin * (1 - 2 * np.sin(spokestep / 4.)), rin))

    spokes = np.zeros(sh)
    contrast = np.min((np.abs(contrast), 1))

    mn = min(spokeint)
    mx = max(spokeint)
    for a in spokeint:
        color = 0.5 - np.abs((a - mn) / (mx - mn) - 0.5)
        spoke = (step(np.real(z * np.exp(-1j*(a + spokestep / 4))), 0)
                 - step(np.real(z * np.exp(-1j*(a - spokestep / 4))), 0))
        spokes += ((spoke * color + 0.5 * np.abs(spoke)) * (1 - contrast)
                   + contrast * np.abs(spoke))

    spokes *= step(r, r0)
    spokes *= step(rlist[-1][0], r)
    for ii in range(len(rlist) - 1):
        a, b = rlist[ii]
        spokes *= (1.0 - rectint(r, a, b))

    if Fast:
        return gaussian_filter(spokes, std)
    else:
        return spokes

def mass_center(A, axes=None, mask=None):
    """
    Calculates mass center of n-dimensional array `A`
    along tuple of axis `axes`.

    Parameters
    ----------
    A : ndarray
        input array

    axes : list, tuple
        Sequence of axes that contribute to distributed mass. If
        ``axes==None``, all axes are considered.

    mask: ndarray, None
        If not None, an array the same shape as A acting as mask

    Returns
    -------
    mass : 1d array
        Center of mass in pixel for each `axis` selected.
    """
    A = np.asarray(A)

    if axes is None:
        axes = tuple(range(1, A.ndim + 1))
    else:
        axes = tuple(np.array(axes) + 1)

    if mask is None:
        return np.sum(A * np.indices(A.shape), axis=axes, dtype=float) / np.sum(A, dtype=float)
    else:
        return np.sum(A * mask * np.indices(A.shape), axis=axes, dtype=float) / np.sum(A * mask, dtype=float)


def radial_distribution(A, radii=None):
    """
    Returns radial mass distribution up to radii in `radii`.

    Parameters
    ----------
    A : ndarray
        input array

    radii : list, tuple
        Sequence of radii to calculate enclosed mass. If `None`,
        the sequence defaults to ``range(1, np.min(A.shape) / 2)``

    Returns
    -------
    radii, masses : list
        Sequence of used `radii` and corresponding integrated mass
        `masses`. Sequences have the same length.

    """
    if radii is None:
        radii = list(range(1, np.min(A.shape) // 2))

    coords = np.indices(A.shape) - np.reshape(mass_center(A),
                                              (A.ndim,) + A.ndim * (1,))

    masses = [np.sum(A * (np.sqrt(np.sum(coords**2, 0)) < r)) for r in radii]

    return radii, masses

def stxm_analysis(storage, probe=None):
    """
    Performs a stxm analysis on a storage using the pods.

    This function is MPI compatible.

    Parameters
    ----------
    storage : ptypy.core.Storage instance
        A :any:`Storage` instance to be analysed

    probe : None, scalar or array

        - If None, picks a probe from the first view's pod
        - If scalar, uses a Gaussian with probe as standard deviation
        - Else: attempts to use passed value directly as 2d-probe

    Returns
    -------
    trans : ndarray
        Transmission of shape ``storage.shape``.

    dpc_row : ndarray
        Differential phase contrast along row-coordinates, i.e. vertical
        direction (y-direction) of shape ``storage.shape``.

    dpc_col : ndarray
        Differential phase contrast along column-coordinates, i.e.
        horizontal direction (x-direction) of shape ``storage.shape``.
    """
    s = storage

    # Prepare buffers
    trans = np.zeros_like(s.data)
    dpc_row = np.zeros_like(s.data)
    dpc_col = np.zeros_like(s.data)
    nrm = np.zeros_like(s.data) + 1e-10

    t2 = 0.
    # Pick a single probe view for preparation purpose:
    v = s.views[0]
    pp = list(v.pods.values())[0].pr_view
    if probe is None:
        pr = np.abs(pp.data) # .sum(0)
    elif np.isscalar(probe):
        x, y = au.grids(pp.shape[-2:])
        pr = np.exp(-(x**2 + y**2) / probe**2)
    else:
        pr = np.asarray(probe)
        assert (pr.shape == pp.shape[-2:]), 'stxm probe has not the same shape as a view to this storage'

    for v in s.views:
        pod = list(v.pods.values())[0]
        if not pod.active:
            continue
        t = pod.diff.sum()
        if t > t2:
            t2=t
        ss = v.slice
        # ss = (v.layer,
        #       slice(v.roi[0,0], v.roi[1,0]),
        #       slice(v.roi[0,1], v.roi[1,1]))
        # bufview = buf[ss]
        m = mass_center(pod.diff) # + 1.
        q = pod.di_view.storage._to_phys(m)
        dpc_row[ss] += q[0] * v.psize[0] * pr * 2 * np.pi / pod.geometry.lz
        dpc_col[ss] += q[1] * v.psize[1] * pr * 2 * np.pi / pod.geometry.lz
        trans[ss] += np.sqrt(t) * pr
        nrm[ss] += pr

    parallel.allreduce(trans)
    parallel.allreduce(dpc_row)
    parallel.allreduce(dpc_col)
    parallel.allreduce(nrm)
    dpc_row /= nrm
    dpc_col /= nrm
    trans /= nrm * np.sqrt(t2)

    return trans, dpc_row, dpc_col

def stxm_init(storage, probe=None):
    """
    Convenience script that performs a STXM analysis for storage
    `storage` given a probe array `probe` and stores result back
    in storage.

    See also
    --------
    stxm_analysis
    """
    trans, dpc_row, dpc_col = stxm_analysis(storage,probe)
    storage.data = trans * np.exp(-1j*phase_from_dpc(dpc_row, dpc_col))

def load_from_ptyr(filename, what='probe', ID=None, layer=None):
    """
    Convenience script to extract data from ``*.ptyr``-file.

    Parameters
    ----------
    filename : str
        Full Path to a ``*.ptyr`` data file. No check on the file suffix
        is done. Any compatible hdf5 formatted file is allowed.
    what : str
        Type of container to retrieve. Only `'probe'` and `'obj'` makes
        sense. Default is `'probe'`
    ID : str
        ID of storage in chosen container. If ``None`` the first stored
        storage is chosen.
    layer : int, optional
        If an integer, the data buffer of chosen storage gets sliced
        with `layer` for its first index.

    Returns
    -------
    data : ndarray
        If `layer` is provided, that layer ``storage,data[layer]``
        will be sliced from the 3d data buffer, else the whole buffer
        ``storage.data`` will be returned.
    """
    from .. import io
    header = io.h5read(filename, 'header')['header']
    if str(header['kind']) == 'fullflat':
        raise NotImplementedError(
            'Loading specific data from flattened dump not yet supported.')
    else:
        if ID is not None:
            address ='content/' + str(what) + '/' + str(ID)
            storage = io.h5read(filename, address)[address]
        else:
            address = 'content/' + str(what)
            conti = io.h5read(filename, address)[address]
            storage = list(conti.values())[0]
        if layer is None:
            return storage['data']
        else:
            return storage['data'][layer]

def phase_from_dpc(dpc_row, dpc_col):
    """
    Implements fourier integration method for two differential quantities.

    Assumes 2 arrays of N-dimensions which contain differential quantities
    in X and Y, i.e. the LAST two dimensions (-2 & -1) in this case.

    Parameters
    ----------
    dpc_row, dpc_col: ndarray
        Differential information along 2nd last dimension
        and last dimension respectively. Must be of the same shape.

    Returns
    -------
    out : ndarray
        Integrated array of same shape as `dpc_row` and `dpc_col`.

    """
    py =- dpc_row
    px =- dpc_col
    sh = px.shape
    sh = np.asarray(sh)
    fac = np.ones_like(sh)
    fac[-2:] = 2
    f = np.zeros(sh * fac, dtype=np.complex)
    c = px + 1j*py
    f[..., :sh[-2], :sh[-1]] = c
    f[..., :sh[-2], sh[-1]:] = c[..., :, ::-1]
    f[..., sh[-2]:, :sh[-1]] = c[...,::-1, :]
    f[..., sh[-2]:, sh[-1]:] = c[...,::-1, ::-1]
    # fft conform grids in the boundaries of [-pi:pi]
    g = au.grids(f.shape, psize=np.pi / np.asarray(f.shape), center='fft')
    g[..., 0, 0] = 1e-6
    qx = g[-2]
    qy = g[-1]
    inv_qc = 1. / (qx + 1j*qy)
    inv_qc[..., 0, 0] = 0
    nf = np.fft.ifft2(np.fft.fft2(f) * inv_qc)

    return np.real(nf[..., :sh[-2], :sh[-1]])

_cxro_server = 'https://henke.lbl.gov'

_cxro_POST_query = ''.join([
    'Material=Enter+Formula',
    '&Formula=%(formula)s&Density=%(density)s&Scan=Energy',
    '&Min=%(emin)s&Max=%(emax)s&Npts=%(npts)s&Output=Text+File'])

def cxro_iref(formula, energy, density=-1, npts=100):
    """\
    Query CXRO database for index of refraction values for a solid.

    Parameters
    ----------
    formula: str
        String representation of the Formula to use.

    energy : float or (float, float)
        Either a single energy (in eV) or the minimum/maximum bounds.

    density : None or float, optional
        Density of the material [g/ccm]. If ``None`` or ``<0`` the
        regular density at ambient temperatures is used.

    npts : int, optional
        Number of points between the min and max energies.

    Returns
    -------
    energy, delta, beta : scalar or vector
        Energy used and the respective `delta` and `beta` values.
    """
    if np.isscalar(energy):
        emin = energy
        emax = energy
        npts = 1
    else:
        emin, emax = energy

    if density is None or density < 0:
        density = -1

    data = cxro_iref.cxro_query % {'formula': formula,
                                   'emin': emin,
                                   'emax': emax,
                                   'npts': npts,
                                   'density': density}

    url = cxro_iref.cxro_server + '/cgi-bin/getdb.pl'
    #u.logger.info('Querying CRXO database...')
    data = data.encode("utf-8")
    req = urllib.request.Request(url)
    response = urllib.request.urlopen(req, data=data)
    t = response.read().decode()
    datafile = t[t.find('/tmp/'):].split('"')[0]

    url = cxro_iref.cxro_server + datafile
    req = urllib.request.Request(url)
    response = urllib.request.urlopen(req)
    data = response.read().decode()

    d = data.split('\n')
    dt = np.array([[float(x) for x in dd.split()] for dd in d[2:] if dd])

    #u.logger.info('done, retrieved: ' +  d[0].strip())
    #print d[0].strip()
    if npts == 1:
        return dt[-1, 0], dt[-1, 1], dt[-1, 2]
    else:
        return dt[:, 0], dt[:, 1], dt[:, 2]

cxro_iref.cxro_server = _cxro_server
cxro_iref.cxro_query = _cxro_POST_query

def remove_hot_pixels(data, size=3, tolerance=3, ignore_edges=False):
    """
    Replaces outlier data points with the median of the surrounding data points.

    Removes outlier data points of a 2D numpy array and replaces them with the
    median of the surrounding data points. Original code (see at the end of
    function) had been published by DanHickstein on stackoverflow.com under
    CC BY-SA license.

    Parameters
    ----------
    data : 2d np.array
        Input 2D numpy array to be corrected.

    size : int
        Size of the window on which the median filter will be applied around
        every data point.

    tolerance: int
        Tolerance multiplied with the standard deviation of the data array
        subtracted by the blurred array (difference array) yields the threshold
        for cutoff.

    ignore_edges : bool
        If True, edges of the array are ignored, which speeds up the code.

    Returns
    -------

    out : 2d array, list
        Returns a 2d np.array where outlier data points have been replaced with
        the median of the surrounding data points and a list containing the
        coordinates of the outlier data points.
    """
    blurred = ndi.median_filter(data, size=size)
    difference = data - blurred
    threshold = tolerance * np.std(difference)

    # Find the hot pixels, but ignore the edges
    hot_pixels = np.nonzero((np.abs(difference[1:-1, 1:-1]) > threshold))
    # Because the first row and first column had been ignored
    hot_pixels = np.array(hot_pixels) + 1

    fixed_image = np.copy(data)
    for y, x in zip(hot_pixels[0], hot_pixels[1]):
        fixed_image[y, x] = blurred[y, x]

    if not ignore_edges:
        height, width = np.shape(data)

        # Now get the pixels on the edges (but not the corners)
        # Left and right sides
        for index in range(1, height - 1):
            # Left side:
            med = np.median(data[index - 1:index + 2, 0:2])
            diff = np.abs(data[index, 0] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[index], [0]]))
                fixed_image[index, 0] = med

            # Right side:
            med = np.median(data[index - 1:index + 2, -2:])
            diff = np.abs(data[index, -1] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[index], [width - 1]]))
                fixed_image[index, -1] = med

        # Then the top and bottom
        for index in range(1, width - 1):
            # Bottom:
            med = np.median(data[0:2, index - 1:index + 2])
            diff = np.abs(data[0, index] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[0], [index]]))
                fixed_image[0,index] = med

            # Top:
            med = np.median(data[-2:, index - 1:index + 2])
            diff = np.abs(data[-1, index] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[height - 1], [index]]))
                fixed_image[-1, index] = med

        #Then the corners

        # Bottom left
        med = np.median(data[0:2, 0:2])
        diff = np.abs(data[0, 0] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[0], [0]]))
            fixed_image[0, 0] = med

        # Bottom right
        med = np.median(data[0:2, -2:])
        diff = np.abs(data[0, -1] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[0], [width - 1]]))
            fixed_image[0, -1] = med

        # Top left
        med = np.median(data[-2:, 0:2])
        diff = np.abs(data[-1, 0] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[height - 1], [0]]))
            fixed_image[-1, 0] = med

        # Top right
        med = np.median(data[-2:, -2:])
        diff = np.abs(data[-1, -1] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[height - 1], [width - 1]]))
            fixed_image[-1, -1] = med

    return fixed_image, hot_pixels

    # Original code
    # def find_outlier_pixels(data,tolerance=3,worry_about_edges=True):
    # This function finds the hot or dead pixels in a 2D dataset.
    # tolerance is the number of standard deviations used to cutoff the hot
    # pixels.
    # If you want to ignore the edges and greatly speed up the code, then set
    # worry_about_edges to False.
    #
    # The function returns a list of hot pixels and also an image with with
    # hot pixels removed
    #
    # from scipy.ndimage import median_filter
    # blurred = median_filter(Z, size=2)
    # difference = data - blurred
    # threshold = 10*np.std(difference)
    #
    # find the hot pixels, but ignore the edges
    # hot_pixels = np.nonzero((np.abs(difference[1:-1,1:-1])>threshold) )
    # because we ignored the first row and first column
    # hot_pixels = np.array(hot_pixels) + 1
    #
    # fixed_image = np.copy(data) #This is the image with the hot pixels removed
    # for y,x in zip(hot_pixels[0],hot_pixels[1]):
    #    fixed_image[y,x]=blurred[y,x]
    #
    # if worry_about_edges == True:
    #    height,width = np.shape(data)
    #
    #    ###Now get the pixels on the edges (but not the corners)###
    #
    #    #left and right sides
    #    for index in range(1,height-1):
    #        #left side:
    #        med  = np.median(data[index-1:index+2,0:2])
    #        diff = np.abs(data[index,0] - med)
    #        if diff>threshold:
    #            hot_pixels = np.hstack(( hot_pixels, [[index],[0]]  ))
    #            fixed_image[index,0] = med
    #
    #        #right side:
    #        med  = np.median(data[index-1:index+2,-2:])
    #        diff = np.abs(data[index,-1] - med)
    #        if diff>threshold:
    #            hot_pixels = np.hstack(( hot_pixels, [[index],[width-1]]  ))
    #            fixed_image[index,-1] = med
    #
    #    #Then the top and bottom
    #    for index in range(1,width-1):
    #        #bottom:
    #        med  = np.median(data[0:2,index-1:index+2])
    #        diff = np.abs(data[0,index] - med)
    #        if diff>threshold:
    #            hot_pixels = np.hstack(( hot_pixels, [[0],[index]]  ))
    #            fixed_image[0,index] = med
    #
    #        #top:
    #        med  = np.median(data[-2:,index-1:index+2])
    #        diff = np.abs(data[-1,index] - med)
    #        if diff>threshold:
    #            hot_pixels = np.hstack(( hot_pixels, [[height-1],[index]]  ))
    #            fixed_image[-1,index] = med
    #
    #    ###Then the corners###
    #
    #    #bottom left
    #    med  = np.median(data[0:2,0:2])
    #    diff = np.abs(data[0,0] - med)
    #    if diff>threshold:
    #        hot_pixels = np.hstack(( hot_pixels, [[0],[0]]  ))
    #        fixed_image[0,0] = med
    #
    #    #bottom right
    #    med  = np.median(data[0:2,-2:])
    #    diff = np.abs(data[0,-1] - med)
    #    if diff>threshold:
    #        hot_pixels = np.hstack(( hot_pixels, [[0],[width-1]]  ))
    #        fixed_image[0,-1] = med
    #
    #    #top left
    #    med  = np.median(data[-2:,0:2])
    #    diff = np.abs(data[-1,0] - med)
    #    if diff>threshold:
    #        hot_pixels = np.hstack(( hot_pixels, [[height-1],[0]]  ))
    #        fixed_image[-1,0] = med
    #
    #    #top right
    #    med  = np.median(data[-2:,-2:])
    #    diff = np.abs(data[-1,-1] - med)
    #    if diff>threshold:
    #        hot_pixels = np.hstack(( hot_pixels, [[height-1],[width-1]]  ))
    #        fixed_image[-1,-1] = med
    #
    # return hot_pixels,fixed_image
