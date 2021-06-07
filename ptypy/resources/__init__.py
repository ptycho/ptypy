import pkg_resources
import numpy as np

flowerfile = pkg_resources.resource_filename(__name__,'flowers.png')
moonfile = pkg_resources.resource_filename(__name__,'moon.png')
treefile = pkg_resources.resource_filename(__name__,'tree.png')

def flower_obj(shape=None):
    from ptypy import utils as u
    import numpy as np
    from matplotlib.image import imread
    
    im = u.rgb2complex(imread(flowerfile))
    if shape is not None:
        sh = u.expect2(shape)
        ish = np.array(im.shape[:2])
        d = sh-ish
        im = u.crop_pad(im,d,axes=[0,1],cen=None,fillpar=0.0,filltype='mirror')

    return im

def tree_obj(shape=None):
    from ptypy import utils as u
    import numpy as np
    from matplotlib.image import imread

    im = 1.0-imread(treefile).astype(float).mean(-1)
    if shape is not None:
        sh = u.expect2(shape)
        ish = np.array(im.shape[:2])
        d = sh-ish
        im = u.crop_pad(im,d,axes=[0,1],cen=None,fillpar=0.0,filltype='mirror')

    return im

def moon_pr(shape=None):
    from ptypy import utils as u
    import numpy as np
    from matplotlib.image import imread
    
    im = u.rgb2complex(imread(moonfile))
    if shape is not None:
        sh = u.expect2(shape)
        ish = np.array(im.shape[:2]).astype(float)
        im = u.zoom(im,sh/ish)

    return im





objects = dict(
    flower = flower_obj,
    tree = tree_obj
)

probes = dict(
    moon = moon_pr
)
