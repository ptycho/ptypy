"""
Plotting utilities.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
import time
import sys
import os
from PIL import Image
import matplotlib as mpl
import matplotlib.cm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#NODISPLAY = (os.getenv("DISPLAY") is None)
#if NODISPLAY:
#    matplotlib.use('agg')

from .verbose import logger
from .array_utils import grids

__all__ = ['pause', 'rmphaseramp', 'plot_storage', 'imsave', 'imload',
           'complex2hsv', 'complex2rgb', 'hsv2rgb', 'rgb2complex', 'rgb2hsv',
           'hsv2complex', 'franzmap','PtyAxis']

# Improved interactive behavior for old versions of matplotlib
try:
    from matplotlib.pyplot import pause
except ImportError:
    import threading
    if matplotlib.get_backend().lower().startswith('qt4'):
        matplotlib.use('QT4Agg')
        mpl_backend = 'qt'
        from PyQt4 import QtGui
        gui_yield_call = QtGui.qApp.processEvents
    elif matplotlib.get_backend().lower().startswith('qt5'):
        matplotlib.use('QT5Agg')
        mpl_backend = 'qt'
        from PyQt5 import QtGui
        gui_yield_call = QtGui.qApp.processEvents
    elif matplotlib.get_backend().lower().startswith('wx'):
        mpl_backend = 'wx'
        import wx
        gui_yield_call = wx.Yield
    elif matplotlib.get_backend().lower().startswith('gtk'):
        mpl_backend = 'gtk'
        import gtk
        def gui_yield_call():
            gtk.gdk.threads_enter()
            while gtk.events_pending():
                gtk.main_iteration(True)
            gtk.gdk.flush()
            gtk.gdk.threads_leave()
    else:
        mpl_backend = None

    if mpl_backend is not None:
        class _Pause(threading.Thread):
            def __init__(self, timeout, message):
                self.message = message
                self.timeout = timeout
                self.ct = True
                threading.Thread.__init__(self)

            def run(self):
                sys.stdout.flush()
                if self.timeout < 0:
                    input(self.message)
                else:
                    if self.message is not None:
                        print(self.message)
                    time.sleep(self.timeout)
                self.ct = False


        def pause(timeout=-1, message=None):
            """\
            Pause the execution of a script while leaving matplotlib figures
            responsive.
            *Gui aware*

            Parameters
            ----------
            timeout : float, optional
                By default, execution is resumed only after hitting return.
                If timeout >= 0, the execution is resumed after timeout seconds.

            message : str, optional
                Message to diplay on terminal while pausing

            """
            if message is None:
                if timeout < 0:
                    message = 'Paused. Hit return to continue.'
            h = _Pause(timeout, message)
            h.start()
            while h.ct:
                gui_yield_call()
                time.sleep(.01)

    else:
        def pause(timeout=-1, message=None):
            """\
            Pause the execution of a script while leaving matplotlib figures
            responsive.
            **Not** *Gui aware*

            Parameters
            ----------
            timeout : float, optional
                By default, execution is resumed only after hitting return.
                If timeout >= 0, the execution is resumed after timeout seconds.

            message : str, optional
                Message to diplay on terminal while pausing

            """
            if timeout < 0:
                if message is None:
                    message = 'Paused. Hit return to continue.'
                input(message)
            else:
                if message is not None:
                    print(message)
                time.sleep(timeout)

# BD: With version 9.1.0 of PIL, _MODE_CONV has been removed, 
#     see here: https://github.com/python-pillow/Pillow/pull/6057
#     can't see a reason why this is still needed, therefore commenting it out
# FIXME: Is this still needed?
# Fix tif import problem
#Image._MODE_CONV['I;16'] = (Image._ENDIAN + 'u2', None)

# Grayscale + alpha should also work
#Image._MODE_CONV['LA'] = (Image._ENDIAN + 'u1', 2)


def complex2hsv(cin, vmin=0., vmax=None):
    """\
    Transforms a complex array into an RGB image,
    mapping phase to hue, amplitude to value and
    keeping maximum saturation.

    Parameters
    ----------
    cin : ndarray
        Complex input. Must be two-dimensional.

    vmin,vmax : float
        Clip amplitude of input into this interval.

    Returns
    -------
    rgb : ndarray
        Three dimensional output.

    See also
    --------
    complex2rgb
    hsv2rgb
    hsv2complex
    """
    # HSV channels
    h = .5*np.angle(cin)/np.pi + .5
    s = np.ones(cin.shape)

    v = abs(cin)
    if vmin is None:
        vmin = v.min()
    if vmax is None:
        vmax = v.max()
    if vmin==vmax:
        v = np.ones_like(v) * v.mean()
        v = v.clip(0.0, 1.0)
    else:
        assert vmin < vmax
        v = (v.clip(vmin, vmax)-vmin)/(vmax-vmin)
    
    return np.asarray((h, s, v))


def complex2rgb(cin, **kwargs):
    """
    Executes `complex2hsv` and then `hsv2rgb`

    See also
    --------
    complex2hsv
    hsv2rgb
    rgb2complex
    """
    return hsv2rgb(complex2hsv(cin, **kwargs))


def hsv2rgb(hsv):
    """\
    HSV (Hue,Saturation,Value) to RGB (Red,Green,Blue) transformation.

    Parameters
    ----------
    hsv : array-like
        Input must be two-dimensional. **First** axis is interpreted
        as hue,saturation,value channels.

    Returns
    -------
    rgb : ndarray
        Three dimensional output. **Last** axis is interpreted as
        red, green, blue channels.

    See also
    --------
    complex2rgb
    complex2hsv
    rgb2hsv
    """
    # HSV channels
    h, s, v = hsv

    i = (6.*h).astype(int)
    f = (6.*h) - i
    p = v*(1. - s)
    q = v*(1. - s*f)
    t = v*(1. - s*(1.-f))
    i0 = (i % 6 == 0)
    i1 = (i == 1)
    i2 = (i == 2)
    i3 = (i == 3)
    i4 = (i == 4)
    i5 = (i == 5)

    rgb = np.zeros(h.shape + (3,), dtype=h.dtype)
    rgb[:, :, 0] = 255*(i0*v + i1*q + i2*p + i3*p + i4*t + i5*v)
    rgb[:, :, 1] = 255*(i0*t + i1*v + i2*v + i3*q + i4*p + i5*p)
    rgb[:, :, 2] = 255*(i0*p + i1*p + i2*t + i3*v + i4*v + i5*q)

    return rgb


def rgb2hsv(rgb):
    """
    Reverse to :any:`hsv2rgb`
    """
    eps = 1e-6
    rgb = np.asarray(rgb).astype(float)
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    v = maxc
    s = (maxc-minc) / (maxc+eps)
    s[maxc <= eps] = 0.0
    rc = (maxc-rgb[:, :, 0]) / (maxc-minc+eps)
    gc = (maxc-rgb[:, :, 1]) / (maxc-minc+eps)
    bc = (maxc-rgb[:, :, 2]) / (maxc-minc+eps)

    h = 4.0+gc-rc
    maxgreen = (rgb[:, :, 1] == maxc)
    h[maxgreen] = 2.0+rc[maxgreen]-bc[maxgreen]
    maxred = (rgb[:, :, 0] == maxc)
    h[maxred] = bc[maxred]-gc[maxred]
    h[minc == maxc] = 0.0
    h = (h/6.0) % 1.0

    return np.asarray((h, s, v))


def hsv2complex(cin):
    """
    Reverse to :any:`complex2hsv`
    """
    h, s, v = cin
    return v * np.exp(np.pi*2j*(h-.5)) / v.max()


def rgb2complex(rgb):
    """
    Reverse to :any:`complex2rgb`
    """
    return hsv2complex(rgb2hsv(rgb))

HSV_to_RGB = hsv2rgb
RGB_to_HSV = rgb2hsv
P1A_to_HSV = complex2hsv
HSV_to_P1A = hsv2complex


def imsave(a, filename=None, vmin=None, vmax=None, cmap=None):
    r"""
    Take array `a` and transform to `PIL.Image` object that may be used
    by `pyplot.imshow` for example. Also save image buffer directly
    without the sometimes unnecessary Gui-frame and overhead.

    Parameters
    ----------
    a : ndarray
        Two dimensional array. Can be complex, in which case the amplitude
        will be optionally clipped by `vmin` and `vmax` if set.

    filename : str, optionsl
        File path to save the image buffer to. Use '\*.png' or '\*.png'
        as image formats.

    vmin,vmax : float, optional
        Value limits ('clipping') to fit the color scale.
        If not set, color scale will span from minimum to maximum value
        in array

    cmap : str, optional
        Name of the colormap for colorencoding.

    Returns
    -------
    im : PIL.Image
        a `PIL.Image` object.

    See also
    --------
    complex2rgb

    Examples
    --------
    >>> from ptypy.utils import imsave
    >>> from matplotlib import pyplot as plt
    >>> from ptypy.resources import flower_obj
    >>> a = flower_obj(512)
    >>> pil = imsave(a)
    >>> plt.imshow(pil)
    >>> plt.show()

    converts array a into, and returns a PIL image and displays it.

    >>> pil = imsave(a, /tmp/moon.png)

    returns the image and also saves it to filename

    >>> imsave(a, vmin=0, vmax=0.5)

    clips the array to values between 0 and 0.5.

    >>> imsave(abs(a), cmap='gray')

    uses a matplotlib colormap with name 'gray'
    """
    if str(cmap) == cmap:
        cmap = mpl.cm.get_cmap(cmap)

    if a.dtype.kind == 'c':
        i = complex2rgb(a, vmin=vmin, vmax=vmax)
        im = Image.fromarray(np.uint8(i), mode='RGB')

    else:
        if vmin is None:
            vmin = a.min()
        if vmax is None:
            vmax = a.max()
        if vmin == vmax:
            vmin, vmax = 0.9 * vmin, 1.1 * vmax
        im = Image.fromarray((255*(a.clip(vmin,vmax)-vmin)/(vmax-vmin)).astype('uint8'))
        if cmap is not None:
            r = im.point(lambda x: int(cmap(x/255.0)[0] * 255))
            g = im.point(lambda x: int(cmap(x/255.0)[1] * 255))
            b = im.point(lambda x: int(cmap(x/255.0)[2] * 255))
            im = Image.merge("RGB", (r, g, b))

    if filename is not None:
        im.save(filename)
    return im


def imload(filename):
    """\
    Load an image and returns a numpy array. *May get deleted*
    """
    return np.array(Image.open(filename))

# Removing it due to DeprecationWarning in Matplotlib
# DeprecationWarning: Passing raw data via parameters data and lut to register_cmap() is deprecated since 3.3 and will become an error two minor releases later. Instead use: register_cmap(cmap=LinearSegmentedColormap(name, data, lut))
# Franz map
# mpl.cm.register_cmap(name='franzmap', data={'red':   ((0.000,   0,    0),
#                                                       (0.350,   0,    0),
#                                                       (0.660,   1,    1),
#                                                       (0.890,   1,    1),
#                                                       (1.000, 0.5,  0.5)),
#                                             'green': ((0.000,   0,    0),
#                                                       (0.120,   0,    0),
#                                                       (0.160,  .2,   .2),
#                                                       (0.375,   1,    1),
#                                                       (0.640,   1,    1),
#                                                       (0.910,   0,    0),
#                                                       (1.000,   0,    0)),
#                                             'blue':  ((0.000,   0,    0),
#                                                       (0.150,   1,    1),
#                                                       (0.340,   1,    1),
#                                                       (0.650,   0,    0),
#                                                       (1.000,   0,    0))}, lut=256)

# Franz Map
franzmap_cm = {'red':   ((0.000,   0,    0),
                                                      (0.350,   0,    0),
                                                      (0.660,   1,    1),
                                                      (0.890,   1,    1),
                                                      (1.000, 0.5,  0.5)),
                                            'green': ((0.000,   0,    0),
                                                      (0.120,   0,    0),
                                                      (0.160,  .2,   .2),
                                                      (0.375,   1,    1),
                                                      (0.640,   1,    1),
                                                      (0.910,   0,    0),
                                                      (1.000,   0,    0)),
                                            'blue':  ((0.000,   0,    0),
                                                      (0.150,   1,    1),
                                                      (0.340,   1,    1),
                                                      (0.650,   0,    0),
                                                      (1.000,   0,    0))}
                                                      
mpl.cm.register_cmap(cmap=LinearSegmentedColormap(name='franzmap', segmentdata=franzmap_cm, N=256))

def franzmap():
    """\
    Set the default colormap to Franz's map and apply to current image if any.
    """
    mpl.pyplot.rc('image', cmap='franzmap')
    im = mpl.pyplot.gci()

    if im is not None:
        im.set_cmap(matplotlib.cm.get_cmap('franzmap'))
    mpl.pyplot.draw_if_interactive()


def rmphaseramp(a, weight=None, return_phaseramp=False):
    """
    Attempts to remove the phase ramp in a two-dimensional complex array
    ``a``.

    Parameters
    ----------
    a : ndarray
        Input image as complex 2D-array.

    weight : ndarray, str, optional
        Pass weighting array or use ``'abs'`` for a modulus-weighted
        phaseramp and ``Non`` for no weights.

    return_phaseramp : bool, optional
        Use True to get also the phaseramp array ``p``.

    Returns
    -------
    out : ndarray
        Modified 2D-array, ``out=a*p``
    p : ndarray, optional
        Phaseramp if ``return_phaseramp = True``, otherwise omitted

    Examples
    --------
    >>> b = rmphaseramp(image)
    >>> b, p = rmphaseramp(image , return_phaseramp=True)
    """

    useweight = True
    if weight is None:
        useweight = False
    elif isinstance(weight,str) and weight == 'abs':
        weight = np.abs(a)

    ph = np.exp(1j*np.angle(a))
    [gx, gy] = np.gradient(ph)
    gx = -np.real(1j*gx/ph)
    gy = -np.real(1j*gy/ph)

    if useweight:
        nrm = weight.sum()
        agx = (gx*weight).sum() / nrm
        agy = (gy*weight).sum() / nrm
    else:
        agx = gx.mean()
        agy = gy.mean()

    (xx, yy) = np.indices(a.shape)
    p = np.exp(-1j*(agx*xx + agy*yy))

    if return_phaseramp:
        return a*p, p
    else:
        return a*p


# FIXME: Is this function ever used?
def plot_data(data, origin=0., psize=1., **kwargs):
    from ..core import Storage
    return plot_storage(Storage(None, 'Sdata', data=data, origin=origin, psize=psize), **kwargs)


def plot_storage(S, fignum=100, modulus='linear', slices=(slice(1), slice(None), slice(None)), si_axes='x', mask=None,
                 **kwargs):
    """\
    Quickly display the data buffer of a :any:`Storage` instance.

    Keyword arguments are the same as :any:`PtyAxis`

    Parameters
    ----------
    S : Storage
        Storage instance

    fignum : int, optional
        Number of the figure.

    slices : tuple of slices or string of numpy index expression, optional
        Determines what part of Storage buffer is displayed, i.e. which
        layers and which region-of-interest. Layers are subplotted
        horizontically next to each other. Figsize is (6,6*layers)

    modulus : str, optional
        One of `sqrt`, `log` or `linear` to apply to modulus of array
        buffer. Useful to reduce dynamic range for diffraction images.

    si_axes : str, optional
        One of 'x','xy','y' or None, determins which axes display
        length units instead of pixel units

    mask : ndarray, scalar or None
        Bool array of valid pixel data for rescaling.

    Returns
    -------
    fig : maplotlib.pyplot.figure

    See also
    --------
    imsave
    :any:`Storage`
    """

    if str(slices) == slices:
        slc = eval('np.index_exp['+slices+']')
    else:
        slc = slices
    im = S.data[slc].copy()
    imsh = im.shape[-2:]

    if np.iscomplexobj(im):
        phase = np.exp(1j*np.pi*np.angle(im))
        channel = 'c'
    else:
        phase = np.real(np.exp(1j*np.pi*np.angle(im)))  # -1 or 1
        channel = 'r'

    if modulus == 'sqrt':
        im = np.sqrt(np.abs(im)).astype(im.dtype)*phase
    elif modulus == 'log':
        im = np.log10(np.abs(im)+1).astype(im.dtype)*phase
    else:
        modulus = 'linear'

    ttl = str(S.ID) + '#%d' + ', ' + modulus + ' scaled'
    y_unit, y_mag, y_num = length_units(S.psize[0]*imsh[0])
    x_unit, x_mag, x_num = length_units(S.psize[1]*imsh[1])

    if im.ndim == 2:
        im = im.reshape((1,)+im.shape)

    layers = im.shape[0]
    fig = plt.figure(fignum, figsize=(6*layers, 5))
    for l in range(layers):
        ax = fig.add_subplot(1, layers, l+1)
        pax = PtyAxis(ax, data=im[l], channel=kwargs.pop('channel', channel), **kwargs)
        pax.set_mask(mask)
        pax.add_colorbar()
        plt.draw()
        pax._update()
        if si_axes is not None and 'x' in si_axes:
            ax.get_position()
            formatter = lambda x, y: pretty_length(S._to_phys((0, x))[1]*x_mag, digits=3)
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(formatter))
            ax.set_xlabel('x [' + x_unit + ']')
        else:
            ax.set_xlabel('x [Pixel]')
        if si_axes is not None and 'y' in si_axes:
            ax.get_position()
            formatter = lambda x, y: pretty_length(S._to_phys((x, 0))[0]*y_mag, digits=3)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(formatter))
            ax.set_ylabel('y [' + y_unit + ']')
        else:
            ax.set_ylabel('y [Pixel]')
        ax.title.set_text(ttl % l)

    plt.draw()
    return fig

class PtyAxis(object):
    """
    Plot environment for matplotlib to allow for a Image plot with color axis,
    potentially of a potentially complex array.
    
    Please note that this class may undergo changes or become obsolete altogether.
    """
    def __init__(self, ax=None, data=None, channel='r', cmap=None, fontsize=8, **kwargs):
        """
        
        Parameters
        ----------
        
        ax : pyplot.axis
            An axis in matplotlib figure. If ``None`` a figure with a single
            axis will be created.
            
        data : numpy.ndarray
            The (complex) twodimensional data to be displayed. 
            
        channel : str
            Choose
            
            - ``'a'`` to plot absolute/modulus value of the data,
            - ``'p'`` to plot phase value of the data,
            - ``'a'`` to plot real value of the data,
            - ``'a'`` to plot imaginary value of the data,
            - ``'a'`` to plot a composite image where phase channel maps to hue and 
              modulus channel maps to brightness of the color.
              
        cmap : str 
            String representation of one of matplotlibs colormaps.
            
        fontsize : int
            Base font size of labels, etc.
            
        Keyword Arguments
        -----------------
        vmin, vmax : float
            Minimum and maximum value for colormap scaling
            
        rmramp : bool
            Remove phase ramp if ``True``, default is ``False``
            
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        self.ax = ax
        self.shape = None
        self.data = None

        if data is not None: self.set_data(data, False)
        # sets shape and data too

        self.remove_phase_ramp = kwargs.get('rmramp', False)
        self.cax = None
        self.cax_aspect = None
        self.cax_width = None
        self.vmin = kwargs.get('vmin')
        self.vmax = kwargs.get('vmax')
        self.mn = None
        self.mx = None
        self.mask = None
        self.fontsize = fontsize
        self.kwargs = kwargs
        self.cmap = None
        self.channel = None
        self.psize = None
        self.set_channel(channel, False)
        self.set_cmap(kwargs.get('cmap', cmap), False)

    def set_psize(self, psize, update=True):
        assert np.isscalar(psize), 'Pixel size must be scalar value'
        self.psize = np.abs(psize)
        if update:
            self._update()

    def set_channel(self, channel, update=True):
        assert channel in ['a', 'c', 'p', 'r', 'i'], \
            'Channel must be either (a)bs, (p)hase, (c)omplex, (r)eal or (i)maginary'
        self.channel = channel
        if update:
            self._update()
            self._update_colorscale()

    def set_cmap(self, cmap, update=True):
        try:
            self.cmap = mpl.cm.get_cmap(cmap)
        except:
            logger.debug("Colormap `%s` not found. Using `gray`" % str(cmap))
            self.cmap = mpl.cm.get_cmap('gray')
        if update:
            self._update()
            self._update_colorscale()

    def set_clims(self, vmin, vmax, update=True):
        self.vmin = vmin
        self.vmax = vmax
        assert vmin < vmax
        if update:
            self._update()

    def set_mask(self, mask, update=True):
        if mask is not None:
            if np.isscalar(mask) and self.shape is not None:
                x, y = grids(self.shape)
                self.mask = (np.sqrt(x**2+y**2) < np.abs(mask))
            else:
                self.mask = mask
                if self.shape is None:
                    self.shape = self.mask.shape
        else:
            self.mask = None
        if update:
            self._update()

    def set_data(self, data, update=True):
        assert data.ndim == 2, 'Data must be two dimensional. It is %d-dimensional' % data.ndim
        self.data = data
        sh = self.shape
        self.shape = self.data.shape

        if update:
            if sh is not None and self.shape != sh:
                self._update(renew_image=True)
            else:
                self._update()

    def _update(self,renew_image=False):
        if str(self.channel) == 'a':
            imdata = np.abs(self.data)
        elif str(self.channel) == 'r':
            imdata = np.real(self.data)
        elif str(self.channel) == 'i':
            imdata = np.imag(self.data)
        elif str(self.channel) == 'p':
            if self.remove_phase_ramp:
                if self.mask is not None:
                    imdata = np.angle(rmphaseramp(self.data, weight=self.mask))
                else:
                    imdata = np.angle(rmphaseramp(self.data))
            else:
                imdata = np.angle(self.data)
        elif str(self.channel) == 'c':
            if self.remove_phase_ramp:
                if self.mask is not None:
                    imdata = rmphaseramp(self.data, weight=self.mask)
                else:
                    imdata = rmphaseramp(self.data)
            else:
                imdata = self.data
        else:
            imdata = np.abs(self.data)

        if self.mask is not None:
            cdata = imdata if str(self.channel) != 'c' else np.abs(self.data)
            self.mx = np.max(cdata[self.mask])
            if self.vmax is None:  # or self.mx<self.vmax:
                mx = self.mx
            else:
                mx = self.vmax
            self.mn = np.min(cdata[self.mask])
            if self.vmin is None:  # or self.mn>self.vmin:
                mn = self.mn
            else:
                mn = self.vmin
        else:
            mn, mx = self.vmin, self.vmax

        pilim = imsave(imdata, cmap=self.cmap, vmin=mn, vmax=mx)
        if not self.ax.images or renew_image:
            self.ax.imshow(pilim, **self.kwargs)
            plt.setp(self.ax.get_xticklabels(), fontsize=self.fontsize)
            plt.setp(self.ax.get_yticklabels(), fontsize=self.fontsize)
            # determine number of points.
            v, h = self.shape
            steps = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000, 3000]
            Nindex = steps[max([v // s <= 4 for s in steps].index(True) - 1, 0)]
            self.ax.yaxis.set_major_locator(mpl.ticker.IndexLocator(Nindex, 0.5))
            Nindex = steps[max([h // s <= 4 for s in steps].index(True) - 1, 0)]
            self.ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(Nindex, 0.5))
        else:
            self.ax.images[0].set_data(pilim)

        self._update_colorbar(mn, mx)

    def _update_colorscale(self, resolution=256):
        if self.cax is None:
            return
        sh = (resolution, int(resolution/self.cax_aspect))
        psize = (1./sh[0], 1./sh[1])
        cax = self.cax
        cax.cla()
        ver, hor = np.indices(sh) * np.asarray(psize).reshape((len(sh),) + len(sh)*(1,))
        if str(self.channel) == 'c':
            comcax = ver * np.exp(2j*np.pi*hor)
            cax.imshow(imsave(comcax), extent=[0, 1, 0, 1], aspect=self.cax_aspect)
            cax.xaxis.set_visible(True)
            cax.set_xticks([0., 1.])
            plt.setp(self.cax.get_xticklabels(), fontsize=self.fontsize)
            plt.setp(self.cax.get_yticklabels(), fontsize=self.fontsize)
        else:
            cax.imshow(ver, cmap=self.cmap, extent=[0, 1, 0, 1], aspect=self.cax_aspect)
            plt.setp(self.cax.get_yticklabels(), fontsize=self.fontsize)
            cax.xaxis.set_visible(False)
        self.cax.invert_yaxis()
        self._update_colorbar()
        plt.draw()

    def _after_resize_event(self, evt):
        self._update_colorbar()

    def _update_colorbar(self, mn=None, mx=None):
        mn = mn if mn is not None else self.mn
        mn = 0 if mn is None else mn
        mx = mx if mx is not None else self.mx
        mx = 1 if mx is None else mx
        if self.cax is None:
            return
        self.cax.dec = np.floor(np.log10(np.abs(mx-mn))) if mx != mn else 0.
        a = self.ax.get_position().bounds
        b = self.cax.get_position().bounds

        self.cax.set_position((b[0], a[1], self.cax_width, a[3]))

        if self.channel == 'c':
            self.cax.xaxis.set_major_locator(mpl.ticker.FixedLocator([0,np.pi, 2*np.pi]))
            self.cax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(['0', r'$\pi$', r'2$\pi$']))
            self.cax.set_xlabel('phase [rad]', fontsize=self.fontsize+2)
            self.cax.xaxis.set_label_position("top")

        locs = np.linspace(0.02, 1., max((int(a[3]*20), 5)))
        self.cax.yaxis.set_major_locator(mpl.ticker.FixedLocator(locs))
        formatter = lambda x, y: pretty_length(((1-x)*(mx-mn)+mn), 3)
        self.cax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(formatter))
        self.cax.set_ylabel('modulus $\\times 10^{%d}$' % int(self.cax.dec), fontsize=self.fontsize+2)
        self.cax.yaxis.set_label_position("right")

    def add_colorbar(self, aspect=10, fraction=0.2, pad=0.02, resolution=256):
        if str(self.channel) == 'c':
            aspect /= 2.
        self.cax_aspect = aspect
        cax, kw = mpl.colorbar.make_axes_gridspec(self.ax, aspect=aspect, fraction=fraction, pad=pad)
        cax.yaxis.tick_right()
        cax.xaxis.tick_top()
        self.cax = cax
        self.cax_width = cax.get_position().width
        self._update_colorscale()


def length_units(number):
    """\
    Doc Todo
    """
    a = np.floor(np.log10(np.abs(number)))
    if a < -6.0:
        unit = 'nm'
        mag = 1e9
    elif a < -3.0:
        unit = 'um'
        mag = 1e6
    elif a < 0.0:
        unit = 'mm'
        mag = 1e3
    elif a < 3.0:
        unit = 'm'
        mag = 1e0
    else:
        unit = 'km'
        mag = 1e-3
    num = number*mag
    return unit, mag, num


def pretty_length(num, digits=3):
    strnum = ("%1." + "%(di)df" % {'di': digits}) % num
    h = strnum.split('.')[0]
    if len(h) >= digits:
        return h
    else:
        return strnum[:digits+1]
