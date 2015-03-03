import pkg_resources

flowerfile = pkg_resources.resource_filename(__name__,'flowers.jpg')
moonfile = pkg_resources.resource_filename(__name__,'moon.jpg')


def flower_obj(shape=None):
    from ptypy import utils as u
    import numpy as np
    
    im = u.HSV_to_P1A(u.RGB_to_HSV(u.imload(flowerfile)))
    if shape is not None:
        sh = u.expect2(shape)
        ish = np.array(im.shape[:2])
        d = sh-ish
        im = u.crop_pad(im,d,axes=[0,1],cen=None,fillpar=0.0,filltype='mirror')
        
    return im

def moon_pr(shape=None):
    from ptypy import utils as u
    import numpy as np
    
    im = u.HSV_to_P1A(u.RGB_to_HSV(u.imload(moonfile)))
    if shape is not None:
        sh = u.expect2(shape)
        ish = np.array(im.shape[:2]).astype(float)
        im = u.zoom(im,sh/ish)
        
    return im





objects = dict(
    flower = flower_obj
)

probes = dict(
    moon = moon_pr
)
