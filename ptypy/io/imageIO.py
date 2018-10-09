# -*- coding: utf-8 -*-
"""\
Wrapper over Python Imaging Library to read images (+ metadata) and write images.
"""
import numpy as np
import glob


# Tentative support for 12-bit tiff. Requires libtiff library
# This trick might be deprecated if using pillow
try:
    import libtiff12bit
except:
    libtiff12bit = None

__all__ = ['imread']


def imread(filename, doglob=None, roi=None):
    """\
        d,meta = imread(filename)
            reads array in image file and returns it as a numpy array, along with metadata (from header).
            d and meta are lists if filename is a list of file names.

        ... = imread(filename, doglob=True)
            reads all matching files if filename contains unix-style wildcards, and returns lists.

        ... = imread(filename, doglob=False)
            ignores wildcards

        ... = imread(filename, doglob=None) [default]
            behaves like doglob=True, except that it returns a list only if filename contains wildcards,
            while doglob=True always returns a list, even if there is only one match.

        ... = imread(filename, roi=(RowFrom, RowTo, ColumnFrom, ColumnTo))
            returns a region of interest (applied on all files if gobbing or if filename is a list)

    """
    import PIL.Image as PIL
    if not isinstance(filename, str):
        # We have a list
        fnames = filename
    else:
        if doglob is None:
            # glob only if there is a wildcard in the filename
            doglob = glob.has_magic(filename)
        if not doglob:
            fnames = [filename]
        else:
            fnames = sorted(glob.glob(filename))
            if not fnames:
                raise IOError('%s : no match.' % filename)
    ldat = []
    lmeta = []
    for f in fnames:
        try:
            im = PIL.open(f)
            meta = readHeader(im)
            dat = np.array(im)
        except IOError:
            # Maybe we have an unsupported tiff...
            if libtiff12bit is None:
                raise
            im = libtiff12bit.TIFF.open(f)
            c = im.GetField('Compression')
            b = im.GetField('BitsPerSample')
            compname = libtiff12bit.define_to_name_map['Compression'][c]
            meta = {'compression': compname[12:].lower(), 'format': 'TIFF', 'mode': ('I;%d' % b)}
            dat = im.read_image()

        meta['filename'] = f
        lmeta.append(meta)
        if roi is not None:
            ldat.append(dat[roi[0]:roi[1], roi[2]:roi[3], ...].copy())
        else:
            ldat.append(dat)
    if doglob:
        dat = ldat
        meta = lmeta
    return dat, meta


def readHeader(im):
    """\
    Reads and parses as much metadata as possible from an image file
    """

    # start with info field
    meta = dict(im.info.items())
    meta['format'] = im.format
    meta['format_description'] = im.format_description
    meta['mode'] = im.mode
    return meta
