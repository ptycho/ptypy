# -*- coding: utf-8 -*-
"""
ptycho - definition of the upper-level class Ptycho.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
import time

from .. import utils as u
from ..utils import parallel
from ..utils.verbose import logger, _
#from .. import experiment
from .. import engines
#from .. import io
from ..io import interaction
from classes import *
from classes import PTYCHO_PREFIX
from paths import Paths
from manager import *

parallel = u.parallel
import data

__all__ = ['Ptycho']

Ptycho_DEFAULT = u.Param(
        verbose_level = 3,      # Verbosity level
        data_type = 'single',   # 'single' or 'double' precision for reconstruction
        data = {},              # Experimental scan information (probably empty at first)
        model = {},          # POD creation rules.
        # The following 4 are now subset of model
        #scans = {},             # Sub-structure that can contain scan-specific parameters for categories 'illumination', 'sample', 'geometry'
        #illumination = {},      # Information about the probe
        #sample = {},            # All information about the object
        #geometry = {},          # Geometry of experiment - most of it provided by data
        paths = {},                # How to load and save
        engines = [u.Param(name='Dummy')],           # Reconstruction algorithms
        interaction = {}, # Client-server communication,
        plotting = {}          # Plotting parameters for a client
    )


class Ptycho(Base):
    """
    Ptycho : A ptychographic data holder and reconstruction manager.
    
    This is the highest level class. It organizes and contains the data,
    manages the reconstruction engines, and interacts with the outside world.
    
    If MPI is enabled, this class acts both as a manager (rank = 0) and 
    a worker (any rank), and most information exists on all processes.
    In principle the only part that is divided between processes is the
    diffraction data.
    
    By default Ptycho is instantiated once per process, but it can also
    be used as a managed container to load past runs. 
    """
    
    DEFAULT = Ptycho_DEFAULT
    _PREFIX = PTYCHO_PREFIX
    
    def __init__(self, pars=None, level=2,**kwargs):
        """
        Ptycho : A ptychographic data holder and reconstruction manager.
        
        Parameters
        ----------
        pars : dict, Param or str
               If dict or Param, the input parameters required for the
               reconstruction. See Ptycho.DEFAULT
               If str, the filename of a past reconstruction to load from.
        level : int,
                Determines how much the Ptycho instance will initialize:
                <=0 : empty ptypy structure
                1 : reads parameters, configures interaction server
                2 : configures Containers, initializes Modelmanager
                >3 : initializes reconstruction engines.
        """
        super(Ptycho,self).__init__(None,'Ptycho')
        
        # abort if we load complete structure
        if level <= 0: 
            return

        # Blank state
        self.p = u.Param(self.DEFAULT)
        
        # Continue with initialization from parameters
        if pars is not None:
            self.p.update(pars)
        
        self.p.update(kwargs)
        
        self._configure()
        if level >=1:
            self.init_structures()
        if level >=2:
            self.init_data()
        if level >=3:
            self.init_engines()
            self.run()
                
    def _configure(self):
        #################################
        # Global logging level
        #################################
        p = self.p
        u.verbose.set_level(p.verbose_level)

        #################################
        # Global data type switch
        #################################

        self.data_type = p.data_type
        assert p.data_type in ['single', 'double']
        self.FType = np.dtype('f' + str(np.dtype(np.typeDict[p.data_type]).itemsize)).type
        self.CType = np.dtype('c' + str(2*np.dtype(np.typeDict[p.data_type]).itemsize)).type
        logger.info(_('Data type', self.data_type))

        #################################
        # Prepare interaction server
        #################################
        if parallel.master:
            # Create the inteaction server
            self.interactor = interaction.Server(p.interaction)
            
            # Start the thread 
            self.interactor.activate()
            
            # Register self as an accessible object for the client
            self.interactor.objects['Ptycho'] = self

        # Check if there is already a runtime container
        if not hasattr(self,'runtime'):
            self.runtime = u.Param()
            
        # Generate all the paths
        self.paths = Paths(self.p.paths,self.runtime)
        
    def init_structures(self):
        """
        Prepare everything for reconstruction.
        """
        
        p = self.p
               
        # Initialize the reconstruction containers
        self.probe = Container(ptycho=self, ID='Cprobe', data_type='complex')
        self.obj = Container(ptycho=self, ID='Cobj', data_type='complex')
        self.exit = Container(ptycho=self, ID='Cexit', data_type='complex')
        self.diff = Container(ptycho=self, ID='Cdiff', data_type='real')
        self.mask = Container(ptycho=self, ID='Cmask', data_type='bool')
       
        ###################################
        # Initialize data sources load data
        ###################################
        
        # Initialize the model manager
        self.modelm = ModelManager(self, p.model)
    
    def init_data(self):
        # Create the data source object, which give diffraction frames one
        # at a time, supporting MPI sharing.
        self.datasource = self.modelm.make_datasource(self.p.data) 
       
        # Load the data. This call creates automatically the scan managers,
        # which create the views and the PODs.
        self.modelm.new_data()
        
        # print stats
        parallel.barrier()
        self.print_stats()

    def init_engines(self):
        ####################################
        # Initialize engines
        ####################################
               
        # Store the engines in a dict
        self.engines = {}
        
        # Store the run labels in a list to ensure precedence is preserved.
        self.run_labels = []
        
        # Loop through p.engines sub-dictionaries
        for run_label, pars in self.p.engines.iteritems():
            # copy common parameters
            engine_pars = self.p.engine.common.copy()
            
            # Identify engine by name
            engine_class = engines.by_name(pars.name)
            
            # update engine type specific parameters
            engine_pars.update(self.p.engine[pars.name])
            
            # update engine instance specific parameters
            engine_pars.update(pars)
            
            # Create instance
            engine = engine_class(self, engine_pars)
            
            # Store info
            self.engines[run_label] = engine
            self.run_labels.append(run_label)
        
    @property
    def pods(self):
        return self._pool['P']
        
    @property
    def containers(self):
        return self._pool['C']
        
      
    def run(self):
        """
        Start the reconstruction and take additionnal 
        commands interactively.
        """
    
        # Time the initialization
        self.runtime.start = time.asctime()
    
        # Check if there is already a runtime info collector
        if self.runtime.get('iter_info') is None:
            self.runtime.iter_info = []
        
        # Run all engines sequentially
        for run_label in self.run_labels:
            
            # Set a new engine
            engine = self.engines[run_label]
            #self.current_engine = engine
            
            # Prepare the engine
            engine.initialize()

            # Start the iteration loop
            while not engine.finished:
                # Check for client requests
                if parallel.master: 
                    self.interactor.process_requests()
                
                parallel.barrier()
                
                # Check for new data
                self.modelm.new_data()
                
                # Last minute preparation before a contiguous block of iterations
                engine.prepare()
                
                # One iteration
                engine.iterate()
                
                # Display runtime information
                if parallel.master: 
                    info = self.runtime.iter_info[-1]
                    logger.info(('Iteration #%(iteration)d of %(engine)s :: Time %(duration).2f \t' % info) + ('Error ' + str(info['error'].sum(0))))
            
            # Done. Let the engine finish up    
            engine.finalize()
    
            # Save
	    # deactivated for now as something fishy happens through MPI 
            #self.save_run()

        # Clean up - if needed.
        
        # Time the initialization
        self.runtime.stop = time.asctime()
        
    @classmethod
    def _from_dict(cls,dct):
        # this method will be called from save_load on linking
        inst = cls(level=0)
        inst.__dict__.update(dct)
        return inst
        
    @classmethod
    def load_run(cls, runfile, load_data=True):
        """
        Load a previous run.
        
        Parameters
        -----------
        runfile : str
                file dump of Ptycho class
        load_data : (True)
                  also load data (thus regenerating pods & views for 
                  'minimal' dump 
        """
        import save_load
        from .. import io
        
        # determine if this is a .pty file
        if not runfile.endswith('.pty'):
            logger.warning('Only ptypy file type allowed for continuing a reconstruction')
            logger.warning('Exiting..')
            return None
        
        logger.info('Creating Ptycho instance from %s' % runfile)
        header = u.Param(io.h5read(runfile,'header')['header'])
        if header['kind']=='minimal':
            logger.info('Found minimal ptypy dump')
            content = u.Param(io.h5read(runfile,'content')['content'])
            
            logger.info('Creating new Ptycho instance')
            P = Ptycho(content.pars,level=1)
            
            logger.info('Attaching probe and object storages')
            for ID,s in content['probe'].items():
                s['owner']=P.probe
                S=Storage._from_dict(s)
            for ID,s in content['obj'].items():
                s['owner']=P.obj
                S=Storage._from_dict(s)
                #S.owner=P.obj
                
            logger.info('Attaching original runtime information')
            P.runtime = content['runtime']
            P.paths.runtime = P.runtime
        
        elif header['kind']=='fullflat':
            P = save_load.link(io.h5read(runfile,'content')['content'])
            
            logger.info('Configuring data types, verbosity and server-client communication')
            P._configure()

            logger.info('Reconfiguring sharing rules')# and loading data')
            P.modelm.sharing_rules = model.parse_model(P.p.model['sharing'],P.modelm.sharing)
            
            logger.info('Regenerating exit waves')
            P.modelm._initialize_exit(P.pods.values())
            """
            logger.info('Attaching datasource')
            P.datasource = P.modelm.make_datasource(P.p.data)
            
            logger.info('Reconfiguring sharing rules and loading data')
            P.modelm.sharing_rules = model.parse_model(P.p.model['sharing'],P.modelm.sharing)
            P.modelm.new_data()
            

            """
        if load_data:
            logger.info('Loading data')
            P.init_data()
        return P
        
    def save_run(self, alt_file=None, kind='minimal',force_overwrite=True):
        """
        Save run to file.
        
        As for now, diffraction / mask data is not stored
        
        Parameters:
        -----------
        alt_file : str
                Alternative filepath, will override io.save_file
        kind : str ('minimal'(default) or 'full_flat')
               type of saving. If 'minimal', only initial paramters,
               probe and object storages and runtime information is stored.
               If 'full_flat', (almost) complete environment
        """
        import save_load
        from .. import io
        
        destfile = self.paths.save_file
        if alt_file is not None: 
            destfile = u.clean_path(alt_file)

        header = {}
        header['kind']=kind
        header['description'] = 'Ptypy .h5 compatible storage format' 
        
        if parallel.master:
            import os
            if os.path.exists(destfile):
                if force_overwrite:
                    logger.warn('Save file exists but will be overwritten (force_overwrite is True)')
                elif not force_overwrite:
                    raise RuntimeError('File %s exists! Operation cancelled.' % destfile)
                elif force_overwrite is None:
                    ans = raw_input('File %s exists! Overwrite? [Y]/N' % destfile)
                    if ans and ans.upper() != 'Y':
                        raise RuntimeError('Operation cancelled by user.') 
            
            if kind == 'fullflat':
                self.interactor.stop()
                logger.info('Deleting references for interactor, datasource and engines.')
                del self.interactor
                del self.datasource
                del self.paths
                try:
                    del self.engines
                    del self.current_engine
                except:
                    pass

                logger.info('Clearing numpy arrays for exit, diff and mask containers.')
                #self.exit.clear()
                for pod in self.pods.values():
                    del pod.exit
                self.diff.clear()
                self.mask.clear()
                logger.info('Unlinking and saving to %s' % destfile)
                content = save_load.unlink(self)
                #io.h5write(destfile,header=header,content=content)
                
            elif kind == 'minimal':
                self.interactor.stop()
                logger.info('Generating copies of probe, object and parameters and runtime')
                minimal = u.Param()
                minimal.probe = {ID : S._to_dict() for ID,S in self.probe.S.items()}
                minimal.obj = {ID : S._to_dict() for ID,S in self.obj.S.items()}
                minimal.pars = self.p.copy()#_to_dict(Recursive=True)
                minimal.runtime =self.runtime.copy()
                content=minimal
                        
            h5opt = io.h5options['UNSUPPORTED']
            io.h5options['UNSUPPORTED'] = 'ignore'
            logger.info('Saving to %s' % destfile)
            io.h5write(destfile,header=header,content=minimal)
            io.h5options['UNSUPPORTED'] = h5opt
        else:
            pass
        # we have to wait for all processes, just in case the script isn't finished after saving
        parallel.barrier()
        return destfile
        
    def print_stats(self):
        """
        Calculates the memrory usage and other info of ptycho instance 
        """
        space=0
        active_pods = sum(1 for pod in self.pods.values() if pod.active)
        info ="------ Process #%d -------\n" % parallel.rank
        info += "%15s : %7d (%7d active)\n" %('Total Pods',len(self.pods.values()), active_pods)
        info_dbg = info
        total = 0
        for ID,C in self.containers.iteritems():
            space,other = C.info()
            info += "Container %5s : %7.2f MB\n" % (ID,space / 1e6)
            info += other 
            total += space
        info += "%15s : %7.2f MB\n" % ('Total memory',total /1e6)
        logger.info(info,extra={'allprocesses':True})
        #logger.debug(info,extra={'allprocesses':True})

    
