import numpy as np
import os
import glob

from .imageIO import imread
from .edfIO import edfread
from .rawIO import rawread
from .h5rw import h5read

__all__ = ['image_read']


def image_read(filename, *args, **kwargs):
    """
    Try to load image data from any file.

    Parameters
    ----------
    filename: str
              Name of the file. If filename contains a wildcard, load all files that match the pattern.
    Returns
    -------
    (image, metadict) or (list of images, list of metadicts)
    """

    use_imread = True
    special_format = ['.raw', '.edf', '.h5']

    if glob.has_magic(filename):
        # Extra check in case the filename's extension is a wildcard
        all_ext = set([os.path.splitext(f)[1].lower() for f in glob.glob(filename)])
        subset = all_ext.intersection(special_format)
        if len(subset) == 1:
            ext = subset.pop()
            use_imread = False
            filename = os.path.splitext(filename)[0] + ext
    else:
        ext = os.path.splitext(filename)[1].lower()
        if ext in special_format:
            use_imread = False
    if use_imread:
        return imread(filename, *args, **kwargs)
    elif ext == '.edf':
        return edfread(filename, *args, **kwargs)
    elif ext == '.raw':
        return rawread(filename, *args, **kwargs)
    elif ext == '.h5':
        h5_image = h5read(filename, *args, **kwargs)
        def look_for_ndarray(d):
            for k, v in d.items():
                if isinstance(v, np.ndarray):
                    return k, v
                elif isinstance(v, dict):
                    out = look_for_ndarray(v)
                    if out is not None:
                        return (k,) + out
                else:
                    pass
            return None
        if isinstance(h5_image, list):
            h5_arrays = []
            h5_metas = []
            for h5s in h5_image:
                h5a = look_for_ndarray(h5s)
                h5_arrays.append(h5a[-1])
                h5_metas.append({'filename': filename, 'path': '/'.join(h5a[0:-1])})
            return h5_arrays, h5_metas
        else:
            h5_array = look_for_ndarray(h5_image)
            return h5_array[-1], {'filename': filename, 'path': '/'.join(h5_array[0:-1])}
    else:
        raise RuntimeError('Unknown file type')
