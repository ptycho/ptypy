# -*- coding: utf-8 -*-
"""\
IO module for handling the file format EDF used by the ESRF beamline id19
Provides read, write and readHeader functions.

translated to python: April 2010, Dieter Hahn

Among other, contains the functions
    readData
    writeData WHICH IS NOT YET IMPLEMENTED
"""
import numpy as np
import glob

from .. import utils

logger = utils.verbose.logger


__all__ = ['edfread']


DATATYPES = {'UnsignedByte': np.uint8,
             'UnsignedShort': np.uint16,
             'SignedInteger': np.uint32,
             'SignedLong' : np.int32,
             'UnsignedInteger': np.uint32,
             'UnsignedInt' : np.uint32,
             'UnsignedLong' : np.uint32,
             'Float' : np.float32,
             'FloatValue' : np.float32,
             'Real' : np.float32,
             'DoubleValue' : np.float64}

def edfread(filename, doglob=None, roi=None):
    """\
        d,meta = edfread(filename)
            reads array in edf file filename and returns it as a numpy array, along with metadata (from header).
            d and meta are lists if filename is a list of file names.

        ... = edfread(filename, doglob=True)
            reads all matching files if filename contains unix-style wildcards, and returns lists.

        ... = edfread(filename, doglob=False)
            ignores wildcards

        ... = edfread(filename, doglob=None) [default]
            behaves like doglob=True, except that it returns a list only if filename contains wildcards,
            while doglob=True always returns a list, even if there is only one match.

        ... = edfread(filename, roi=(RowFrom, RowTo, ColumnFrom, ColumnTo))
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
        logger.info('Reading "%s"' % f)
        meta = readHeader(f)
        dat = _readImage(f,DATATYPES[meta["DataType"]], meta["Dim_1"],meta["Dim_2"],meta["headerlength"])
        meta['filename'] = f
        lmeta.append(meta)
        if meta["DataType"]=="Float" or meta["DataType"]=="FloatValue":
            if meta["ByteOrder"]=="HighByteFirst":
                logger.debug("EDF File ByteOrder = HighByteFirst. Converting array to Big Endian")
                dat=dat.newbyteorder("B")
            else:
                dat=dat.newbyteorder("L")
                logger.debug("EDF File ByteOrder = Not HighByteFirst. Converting array to Little Endian")

        if roi is not None:
            ldat.append(dat[roi[0]:roi[1],roi[2]:roi[3]].copy())
        else:
            ldat.append(dat)
    if doglob:
        dat = ldat
        meta = lmeta
    return dat,meta


def readData(filenameprefix,imgstart=0,imgnumber = 1,xi = 0, xf = 0, bin_fact = 1,rowFrom = 0, rowTo = 0,multiple = 1):
    """\
        Reads image data from .edf file.

        [dat, meta] = readData(path+filename, imgstart=0,imgnumber = 1,xi = 0, xf = 0, bin = 1,rowFrom = 0, rowTo = 0,multiple = 1)

        with multiple = 1 (default, corresponds to ESRF id19 file format):
            Reads in 'imgnumber' image files beginning from 'imgstart' and stores the data in a
            list of numpy.array(ydim,xdim). Stores metadata in dictionaries which are themselves
            stored in a list.

        with multiple = 0
            Reads filenameprefix.edf

        returns: [dat, meta]:
                                datm = list of 2d arrays containing image data
                                meta = list of dictionaries containing metadata for each image


        example:
        [dat, meta] = readData('some_prefix',imgstart=4,imgnumber = 3, multiple = 1) reads 3 images:
        some_prefix_0004_0000.edf
        some_prefix_0004_0001.edf
        some_prefix_0004_0002.edf
        and stores their respective data and headers in dat and meta

        [dat, meta] = readData('other_prefix', multiple = 0) reads
        other_prefix.edf
    """
    # initialization of return values
    datm = []
    meta = []
    if multiple == 1:
        headerlength=2048
        if (rowTo < rowFrom and rowTo != 0):
            raise ValueError('The last row has to be equal or larger than the first row.\n')
        for imgnum in range(imgnumber):
            filename = filenameprefix + '_' + utils.num2str(imgstart,'%04d') + '_' + utils.num2str(imgnum,'%04d') + '.edf'
            logger.info('loading %s' % filename)
            # needed from header are meta[i]["counter"]["mon"] and meta[i]["count_time"]
            meta.append(readHeader(filename,headerlength=headerlength))
            col_beg = meta[imgnum]["col_beg"]
            col_end = meta[imgnum]["col_end"]
            no_of_cols = col_end - col_beg +1
            if no_of_cols < 1:
                raise ValueError('Invalid number of columns extracted from edf file header\n')
            # load the data ('dat' will be np.array)
            # TODO: implement endianness
            if rowTo > 0:
                # read only part of image
                dat = _readImage(filename,DATATYPES[meta[imgnum]["DataType"]], meta[imgnum]["Dim_1"],meta[imgnum]["Dim_2"],headerlength=headerlength)
            else:
                # read whole image
                dat = _readImage(filename,DATATYPES[meta[imgnum]["DataType"]], meta[imgnum]["Dim_1"],meta[imgnum]["Dim_2"],headerlength=headerlength)
                # bin and normalize the data (should just always append an np.array to the list 'datm')
            if bin_fact == 1 or bin_fact == 0:
                if xf == 0: xf = dat.shape[1]
                if rowTo == 0: rowTo = dat.shape[0]
                datm.append(dat[rowFrom:rowTo,xi:xf].copy())
            else:
                import scipy
                xdim = np.floor(np.size(dat,0)/bin_fact); ydim = np.floor(np.size(dat,1)/bin_fact)
                datm.append(scipy.misc.pilutil.imresize(dat,(xdim, ydim)))
    else:
        headerlength=1024
        # Add the extension if it is not there.
        if filenameprefix[-4:] == '.edf':
            filename = filenameprefix
        else:
            filename = filenameprefix + '.edf'
        logger.info('loading %s' % filename)
        # needed from header are meta[i]["counter"]["mon"] and meta[i]["count_time"]
        meta.append(readHeader(filename,headerlength=headerlength))
        # load the data ('dat' will be np.array)
        # TODO: implement endianness
        if rowTo > 0:
            #dat = edfreadl(filename,range(1,no_of_cols), range(rowFrom,rowTo))
            # read only part of image
            dat = _readImage(filename,DATATYPES[meta[0]["DataType"]], meta[0]["Dim_1"],meta[0]["Dim_2"],headerlength=headerlength)
        else:
            # read whole image
            dat = _readImage(filename,DATATYPES[meta[0]["DataType"]], meta[0]["Dim_1"],meta[0]["Dim_2"],headerlength=headerlength)
        # bin and normalize the data (should just always append an np.array to the list 'datm')
        if bin == 1 or bin == 0:
            datm.append(dat)
        else:
            import scipy
            xdim = np.floor(np.size(dat,0)/bin); ydim = np.floor(np.size(dat,1)/bin)
            datm.append(scipy.misc.pilutil.imresize(dat,(xdim, ydim)))

    return (datm,meta)

def _readImage(filename,dtype,width,height,headerlength=2048):
    """Reads the actual image data from given file.
    Returns numpy.ndarray"""
    f1 = open(filename)
    f1.seek(headerlength)
    im = np.fromfile(f1,dtype).reshape([height,width])
    f1.close()
    return im

def writeData():
    """ not yet implemented """
    pass

def readHeader(filename, headerlength=None):
    """Reads and parses  the metadata contained in the header of an .edf file.
    Returns dictionary."""

    with open(filename,'rb') as f:
        if f.read(1) != b'{':
            raise RuntimeError('File "%s" does not seem to be edf format.' % filename)
        if headerlength is None:
            f.seek(2)
            s = f.read(10*1024)
            headerlength = s.find(b'}\n') + 4
            s = s[:headerlength-2]
        else:
            f.seek(2)
            s = f.read(headerlength-2)
            
    # Encoding s as utf-8
    s = str(s,'utf-8')
    # split read string into a list of 2-element lists
    hlist = [elem.split("=") for elem in s.replace("\n","").split(";")]
    hlist = [[convertStr(elem.strip()) for elem in elem2] for elem2 in hlist]
    # convert to dictionary
    hdict = dict(hlist[0:-1])
    # convert counter and motor settings in separate dictionaries inside hdict
    if 'counter_mne' in hdict:
        hdict["counter"] = dict(zip(hdict["counter_mne"].split(" "),[convertStr(elem) for elem in hdict["counter_pos"].split(" ")]))
    if 'motor_mne' in hdict:
        hdict["motor"] = dict(zip(hdict["motor_mne"].split(" "),[convertStr(elem) for elem in hdict["motor_pos"].split(" ")]))
    # add header length in meta-data
    hdict["headerlength"] = headerlength
    # add local filename in meta-data
    hdict["local_filename"] = filename
    return hdict

def convertStr(s):
    try:
        ret = int(s)
    except ValueError:
        # then try float.
        try:
            ret = float(s)
        except ValueError:
            # neither int nor float
            ret = s
    return ret
