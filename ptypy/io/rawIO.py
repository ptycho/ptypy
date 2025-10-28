#-*- coding: utf-8 -*-
"""
read .raw images from FLI camera
Created in Nov 2013
TODO: add masking function to mask out hot/dead pixels of detector
@author: Bjoern Enders
"""
import glob
import numpy as np
from ptypy.utils.verbose import log

__all__ = ['rawread']


DATATYPES = {'UnsignedByte': np.uint8,
             'UnsignedShort': np.uint16,
             'SignedInteger': np.uint32,
             'UnsignedInteger': np.uint32,
             'UnsignedInt' : np.uint32,
             'UnsignedLong' : np.uint32,
             'Float' : np.float32,
             'FloatValue' : np.float32,
             'Real' : np.float32,
             'DoubleValue' : np.float64}

def rawread(filename, doglob=None, roi=None):
    """\
        d,meta = rawread(filename)
            reads array in CDF file filename and returns it as a numpy array, along with metadata (from header).
            d and meta are lists if filename is a list of file names.

        ... = rawread(filename, doglob=True)
            reads all matching files if filename contains unix-style wildcards, and returns lists.

        ... = rawread(filename, doglob=False)
            ignores wildcards

        ... = rawread(filename, doglob=None) [default]
            behaves like doglob=True, except that it returns a list only if filename contains wildcards,
            while doglob=True always returns a list, even if there is only one match.

        ... = rawread(filename, roi=(RowFrom, RowTo, ColumnFrom, ColumnTo))
            returns a region of interest (applied on all files if gobbing or if filename is a list)

    """
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
        log(3, 'Reading "%s"' % f)
        dat,meta = _read(f,np.uint16)
        meta['filename'] = f
        lmeta.append(meta)
        if roi is not None:
            ldat.append(dat[roi[0]:roi[1],roi[2]:roi[3]].copy())
        else:
            ldat.append(dat)
    if doglob:
        dat = ldat
        meta = lmeta
    return dat,meta

def _read(filename,dtype):
    f=open(filename)
    header = []
    header.append(f.readline())

    while 'EOH' not in header[-1]:
        header.append(f.readline())

    meta = _interpret_header(header)
    rows = meta['rows']
    cols = meta['cols']
    f.seek(-rows*cols*2,2)
    data = np.fromfile(f,dtype)
    data = data.reshape((rows,cols))

    return data,meta

def _interpret_header(header):
    """
    This is special for FLI server I guess.
    """
    #print header
    meta=dict(
    cam_server_version = header[0].split()[2:],
    ccd_model = header[1].split()[2:],
    rows = np.int(header[2].split()[2]),
    cols = np.int(header[3].split()[2]),
    exp_time = np.float(header[6].split()[2]),
    exp_time_theory = np.float(header[4].split()[2]),
    exp_time_measured = np.float(header[5].split()[2]),
    timestamp_string = header[7].split()[2:],
    timestamp_integer = np.int(header[9].split()[2]),
    monitorcounts = np.int(header[10].split()[2]),
    binning_rows = np.int(header[11].split()[2]),
    binning_cols = np.int(header[12].split()[2]),
    roi_visible = [np.int(x) for x in header[13].split()[2:]],
    roi_set = [np.int(x) for x in header[14].split()[2:]],
    raw_header = header
    )
    return meta

