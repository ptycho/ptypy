#!/usr/bin/python

import numpy as np
import re, glob, os
from ptypy.core import data
from ptypy import utils as u
from ptypy import io

try:
    from Tkinter import Tk
    from tkFileDialog import askopenfilename, askopenfilenames
    print("Please, load the first frame of the ptychography experiment...")
    Tk().withdraw()
    pathfilename = askopenfilename(initialdir='.', title='Please, load the first frame of the ptychography experiment...')      
except ImportError:
    print('Please, give the full path for the first frame of the ptychography experiment')
    pathfilename = raw_input('Path:')
    
filename = pathfilename.rsplit('/')[-1]
path = pathfilename[:pathfilename.find(pathfilename.rsplit('/')[-1])]

default_recipe = u.Param(
    first_frame = filename,
    base_path = path,
)

class ID16Scan(data.PtyScan):
    """
    Class ID16Scan
    Data preparation for far-field ptychography experiments at ID16A beamline - ESRF using FReLoN camera
    First version by B. Enders (12/05/2015)
    Modifications by J. C. da Silva (30/05/2015)
    """
    
    def __init__(self, pars=None):
        super(ID16Scan,self).__init__(pars)
        r = self.info.recipe
        # filename analysis
        body,ext=os.path.splitext(os.path.expanduser(r.base_path+r.first_frame))
        sbody = re.sub('\d+$','',body)
        num = re.sub(sbody,'',body)
        # search string for glob
        self.frame_wcard = re.sub('\d+$','*',body)+ext
        # format string for load
        self.frame_format = sbody + '%0'+str(len(num))+'d'+ext
        # count the number of available frames
        self.num_frames = len(glob.glob(self.frame_wcard))
        
    def _frame_to_index(self,fname):
        body,ext = os.path.splitext(os.path.split(fname)[-1])
        return int(re.sub(re.sub('\d+$','',body),'',body))-1
    
    def _index_to_frame(self,index):
        return self.frame_format % (index+1)
    
    def _load_dark(self):
        r = self.info.recipe
        print('Loading the dark files...')
        darklist = []
        for ff in sorted(glob.glob(r.base_path+'dark*.edf')):
            d,dheader = io.image_read(ff)
            darklist.append(d)
        print('Averaging the dark files...')
        darkavg = np.array(np.squeeze(darklist)).mean(axis=0)
        return darkavg

    def load(self,indices):
        raw = {}
        pos = {}
        weights = {}
        darkavg = self._load_dark()
        for idx in indices:
            r,header = io.image_read(self._index_to_frame(idx))
            img1 = r-darkavg
            raw[idx] = img1
            pos[idx] = (header['motor']['spy']*1e-6,header['motor']['spz']*1e-6)
        return raw,pos,{}
        
pars = dict(
    label=None,   # label will be set internally 
    version='0.2',
    shape=(700,700),
    psize=9.552e-6,
    energy=17.05,
    center=None,
    distance = 1.2,
    dfile = filename[:filename.find('.')][:-4].lower()+'.ptyd',#'siemensstar30s.ptyd',  # filename (e.g. 'foo.ptyd')
    chunk_format='.chunk%02d',  # Format for chunk file appendix.
    save = 'append',  # None, 'merge', 'append', 'extlink'
    auto_center = None,  # False: no automatic center,None only  if center is None, True it will be enforced   
    load_parallel = 'data',  # None, 'data', 'common', 'all'
    rebin = 2,#None,  # rebin diffraction data
    orientation = (True,True,False),  # None,int or 3-tuple switch, actions are (transpose, invert rows, invert cols)
    min_frames = 1,  # minimum number of frames of one chunk if not at end of scan
    positions_theory = None,  # Theoretical position list (This input parameter may get deprecated)
    num_frames = None, # Total number of frames to be prepared
    recipe = default_recipe,
)
u.verbose.set_level(3)
IS = ID16Scan(pars)
IS.initialize()
IS.auto(400)
