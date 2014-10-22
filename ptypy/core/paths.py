# -*- coding: utf-8 -*-
"""
Path manager

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import sys
import os

# for solo use ##########
if __name__ == "__main__":
    from ptypy import utils as u
    from ptypy.utils.verbose import logger

else:
# for in package use #####
    from .. import utils as u
    from ..utils.verbose import logger
    
__all__=['Paths']

DEFAULT=u.Param()
DEFAULT.base_dir = "./"    # (03) Relative base path for all other paths
DEFAULT.run = None                                   # (04) Name of reconstruction run
DEFAULT.plot_dir = "plots/%(run)s/"                  # (06) directory to dump plot images
DEFAULT.plot_file = "%(run)s_%(engine)s_%(iteration)04d.png"# (07) filename for dumping plots
DEFAULT.plot_interval = 2                       # (08) iteration interval for dumping plots
DEFAULT.save_dir = "recons/%(run)s/"                 # (10) directory to save final reconstruction
DEFAULT.save_file = "test.pty" #"%(run)s_%(algorithm)s_%(it)04d.h5"# (11) filename for saving 
DEFAULT.dump_dir = "dumps/%(run)s/"                  # (12) directory to save intermediate results
DEFAULT.dump_file = "%(run)s_%(engine)s_%(iteration)04d.pty"# (13) 
DEFAULT.data_dir = "analysis/%(run)s/"
DEFAULT.data_file = "%(label)s.h5"
# runtime parameters
DEFAULT.engine = "Dummy"
DEFAULT.iteration = 0
DEFAULT.args = ""

class Paths(object):

    def __init__(self,pars=None,runtime=None,make_dirs=True):
        self.runtime = runtime
        self.p=DEFAULT.copy()
        if pars is not None:
            self.p.update(pars)
        
        if self.p.run is None:
            self.p.run = os.path.split(sys.argv[0])[1].split('.')[0]
        
        self.p.args = '#'.join(sys.argv[1:]) if len(sys.argv) != 0 else ''
        self.make_dirs = make_dirs
        
        sep = os.path.sep
        if not self.p.base_dir.endswith(sep):
            self.p.base_dir+=sep
            
        for key in ['plot','save','dump','data']:
            d = key+'_dir'
            if not self.p[d].startswith(sep):
                #append base
                self.p[d]=self.p.base_dir+self.p[d]
                
        for key in ['plot','save','dump','data']:
            f = key+'_file'
            if not self.p[f].startswith(sep):
                # append predir
                self.p[f]=self.p[key+'_dir']+self.p[f]

    @property
    def dump_file(self):
                    
        return self.get_path(self.p.dump_file)
        
    @property
    def save_file(self):
                    
        return self.get_path(self.p.save_file)
        
    @property
    def plot_file(self):
                    
        return self.get_path(self.p.plot_file)
        
    def get_data_file(self,**kwargs):
        
        self.p.update(**kwargs)
        return self.get_path(self.p.data_file)
        
    def get_path(self,path):
        try:
            d = dict(self.runtime.iter_info[-1])
            self.p.update(d)
        except:
            logger.debug('Receiving runtime info for dumping/saving failed')
        
        path = os.path.abspath(os.path.expanduser(path % self.p))

        return path
        
############
# TESTING ##
############

if __name__ == "__main__":
    pa = Paths()
    print pa.save_file
    print pa.dump_file
    print pa.p
