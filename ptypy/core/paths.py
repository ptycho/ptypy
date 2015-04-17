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
DEFAULT.home = "./"    # (03) Relative base path for all other paths
DEFAULT.plots = "plots/%(run)s/%(run)s_%(engine)s_%(iteration)04d.png"# (07) filename for dumping plots
DEFAULT.recons = "recons/%(run)s/%(run)s_%(engine)s.ptyr"                 # (10) directory to save final reconstruction
DEFAULT.autosave = "dumps/%(run)s/%(run)s_%(engine)s_%(iteration)04d.ptyr"                  # (12) directory to save intermediate results
DEFAULT.movie = "plots/%(run)s/%(run)s_%(engine)s.mpg"# (13) 
DEFAULT.data = "analysis/%(run)s/%(label)s.ptyd"
# runtime parameters
DEFAULT.run = None                                   # (04) Name of reconstruction run
DEFAULT.engine = "Dummy"
DEFAULT.iteration = 0
DEFAULT.args = ""

class Paths(object):
    """
    Path managing class
    """
    DEFAULT = DEFAULT
    def __init__(self,pars=None,runtime=None,make_dirs=True):
        """
        Parameters
        ----------
        pars : Param or dict
            Parameter set. See :any:`DEFAULT`
            
        runtime : dict
            Optional runtime dictionary for dynamic file names.
            
        makedirs : bool
            Create directories if they do not exist already.
        """
        self.runtime = runtime
        self.p=DEFAULT.copy()
        if pars is not None:
            self.p.update(pars)
        
        if self.p.run is None:
            self.p.run = os.path.split(sys.argv[0])[1].split('.')[0]
        
        self.p.args = '#'.join(sys.argv[1:]) if len(sys.argv) != 0 else ''
        self.make_dirs = make_dirs
        
        sep = os.path.sep
        if not self.p.home.endswith(sep):
            self.p.home+=sep
        
        """
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
        """
    
    @property
    def auto_file(self):
        """ File path for autosave file """
        return self.get_path(self.p.autosave)
    
    @property
    def recon_file(self):
        """ File path for reconstruction file """           
        return self.get_path(self.p.recons)
    
    @property
    def plot_file(self):
        """ 
        File path for plot file 
        """
        return self.get_path(self.p.plots)
    
    def get_data_file(self,**kwargs):
        
        self.p.update(**kwargs)
        return self.get_path(self.p.data)
    
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
    print pa.auto_file
    print pa.plot_file
    print pa.recon_file
    print pa.p
