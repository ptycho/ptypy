import numpy as np

def flowers(shape=None):
    from ptypy import utils as u
    im = u.HSV_to_P1A(u.RGB_to_HSV(u.imload(flowers.jpg)))
    if shape is not None:
        sh = u.expect2(shape)
        ish = np.array(im.shape[:2])
        d = ish-sh
        im = u.crop_pad(im,d,axes=[0,1],cen=None,fillpar=0.0,filltype='mirror')

    return im
