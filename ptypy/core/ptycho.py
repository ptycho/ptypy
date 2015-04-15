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
from ..utils.verbose import logger, _, report
from .. import engines
from ..io import interaction
from classes import *
from classes import PTYCHO_PREFIX
import paths 
from manager import *
from . import model

parallel = u.parallel

__all__ = ['Ptycho']

DEFAULT_plotclient = u.Param(
    active = True,
    interval = 1,
    layout = {},
    dump = True,
    dump_interval = None,
    make_movie = True,
)

DEFAULT_autosave = u.Param(
    interval = 10,  # if None or False : no saving else save with this given interval
    probes = None,  # list of probe IDs for autosaving, if None save all [Not implemented]
    objects = None, # list of object IDs for autosaving, if None, save all [Not implemented]
)



DEFAULT_ptycho = u.Param(
        verbose_level = 3,      # Verbosity level
        data_type = 'single',   # 'single' or 'double' precision for reconstruction
        dry_run = False,        # do actually nothing if True [not implemented]
        autosave = DEFAULT_autosave,        # (None or False, int) If an integer, specifies autosave interval, if None or False: no autosave
        scan = u.Param(),          # POD creation rules.
        scans=u.Param(),
        paths = paths.DEFAULT.copy(),                # How to load and save
        engines = {},           # Reconstruction algorithms
        engine = engines.DEFAULTS.copy(),
        interaction = {}, # Client-server communication,
        plotclient = DEFAULT_plotclient   # Plotting parameters for a client
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
    
    Attributes
    ----------
    p : Param
        Internal Parameters. Stucture like `DEFAULT`
    
    CType,FType : type
        numpy dtype for arrays. `FType` is for data, i.e. real-valued
        arrays, `CType` is for complex-valued arrays
        
    interactor : ptypy.io.interaction.Server
        ZeroMQ interaction server for communication with e.g. plotting
        clients
    
    runtime : Param
        Runtime information, e.g. errors, iteration etc.
        
    paths : ptypy.core.path.Paths
        File paths
    
    modelm : ModelManager
        THE managing instance for :any:`POD`, :any:`View` and 
        :any:`Geo` instances
        
    probe,obj,exit,diff,mask : Container
        Container instances for illuminations, samples, exit waves,
        diffraction data and detector masks / weights
    """
    
    DEFAULT = DEFAULT_ptycho    
    _PREFIX = PTYCHO_PREFIX
    
    def __init__(self, pars=None, level=2,**kwargs):
        """        
        Parameters
        ----------
        pars : Param
            The input parameters required for the
            reconstruction. See :any:`DEFAULT`
        
        level : int
            Determines how much is initialized.
            
            - <=0 : empty ptypy structure
            - 1 : reads parameters, configures interaction server, 
                  see :py:meth:`init_structures`
            - 2 : also configures Containers, initializes Modelmanager
                  see :py:meth:`init_data`
            - 3 : also initializes reconstruction engines
                  see :py:meth:`init_engines`
            - >=4 : also and starts reconstruction
                    see :py:meth:`run`
        """
        super(Ptycho,self).__init__(None,'Ptycho')
        
        # abort if we load complete structure
        if level <= 0: 
            return

        # Blank state
        self.p = self.DEFAULT.copy(depth=99)
        
        # Continue with initialization from parameters
        if pars is not None:
            self.p.update(pars)
        
        # that may be a little dangerous
        self.p.update(kwargs)
        
        self._configure()
        if level >=1:
            self.init_structures()
        if level >=2:
            self.init_data()
        if level >=3:
            self.init_engines()
        if level >=4:
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
        if parallel.master and p.interaction is not None:
            # Create the inteaction server
            self.interactor = interaction.Server(p.interaction)
            
            # Register self as an accessible object for the client
            self.interactor.objects['Ptycho'] = self

            # Start the thread
            self.interactor.activate()
        
            # inform the audience
            logger.info('Started interaction server with the following parameters:\n'+report(self.interactor.p))
        else:
            # no interaction wanted
            self.interactor = None
            
        # Check if there is already a runtime container
        if not hasattr(self, 'runtime'):
            self.runtime = u.Param()
            
        # Generate all the paths
        self.paths = paths.Paths(self.p.paths,self.runtime)
        
    def init_structures(self):
        """
        Called on __init__ if ``level>=1``.
        
        Prepare everything for reconstruction. Creates attributes
        :py:attr:`modelm` and the containers :py:attr:`probe` for 
        illumination, :py:attr:`obj` for the samples, :py:attr:`exit` for
        the exit waves, :py:attr:`diff` for diffraction data and 
        :py:attr:`mask` for detectors masks
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
        self.modelm = ModelManager(self, p.scan)
    
    def init_data(self, print_stats=True):
        """
        Called on __init__ if ``level>=2``.
        
        Creates a datasource and calls for :py:meth`ModelManager.new_data()`
        Prints statistics on the ptypy structure if ``print_stats=True``
        """
        # Create the data source object, which give diffraction frames one
        # at a time, supporting MPI sharing.
        self.datasource = self.modelm.make_datasource() 
       
        # Load the data. This call creates automatically the scan managers,
        # which create the views and the PODs.
        self.modelm.new_data()
        
        # print stats
        parallel.barrier()
        if print_stats:
            self.print_stats()

    def init_engines(self):
        """
        Initialize engines from paramters. Sets :py:attr:`engines`
        """
               
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
        """ Dict of all :any:`POD` instances in the pool of self """
        return self._pool.get('P', {})
        
    @property
    def containers(self):
        """ Dict of all :any:`Container` instances in the pool of self """
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
        
        # Note when the last autosave was carried out
        if self.runtime.get('last_save') is None:
            self.runtime.last_save = 0
        
        # maybe not needed
        if self.runtime.get('last_plot') is None:
            self.runtime.last_plot = 0
            
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
                if parallel.master and self.interactor is not None: 
                    self.interactor.process_requests()
                
                parallel.barrier()
                
                # Check for new data
                self.modelm.new_data()
                
                # Last minute preparation before a contiguous block of iterations
                engine.prepare()
                
                # One iteration
                engine.iterate()
                
                # Display runtime information and do saving
                if parallel.master: 
                    info = self.runtime.iter_info[-1]
                    # calculate Error:
                    err = np.array(info['error'].values()).mean(0)
                    logger.info('Iteration #%(iteration)d of %(engine)s :: Time %(duration).2f' % info) 
                    logger.info('Errors :: Fourier %.2e, Photons %.2e, Exit %.2e' % tuple(err) )
                    
                if self.p.autosave is not None and self.p.autosave.interval > 1:
                    if engine.curiter >= self.runtime.last_save + self.p.autosave.interval:
                        auto = self.paths.auto_file
                        logger.info('----- Autosaving -----')
                        self.save_run(auto,'dump')
                        self.runtime.last_save = engine.curiter
                        logger.info('----------------------')
                        
                parallel.barrier()
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
        ----------
        runfile : str
                file dump of Ptycho class
        load_data : bool
                If `True` also load data (thus regenerating pods & views 
                for 'minimal' dump
        
        Returns
        -------
        P : Ptycho
            Ptycho instance with ``level==2``
        """
        import save_load
        from .. import io
        
        # determine if this is a .pty file
        # FIXME: do not rely on ".pty" extension.
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
        
        Parameters
        ----------
        alt_file : str
            Alternative filepath, will override io.save_file
            
        kind : str
            Type of saving, one of:
            
                - 'minimal', only initial parameters, probe and object 
                  storages and runtime information is saved.
                - 'full_flat', (almost) complete environment
               
        """
        import save_load
        from .. import io
        
        destfile = self.paths.recon_file
        if alt_file is not None and parallel.master: 
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
                
            elif kind == 'dump':
                #if self.interactor is not None:
                #    self.interactor.stop()
                logger.info('Generating copies of probe, object and parameters and runtime')
                dump = u.Param()
                dump.probe = {ID : S._to_dict() for ID,S in self.probe.S.items()}
                dump.obj = {ID : S._to_dict() for ID,S in self.obj.S.items()}
                dump.pars = self.p.copy()#_to_dict(Recursive=True)
                dump.runtime = self.runtime.copy()
                # discard some bits of runtime to save space
                dump.runtime.iter_info = [self.runtime.iter_info[-1]]

                content=dump
                
            elif kind == 'minimal':
                #if self.interactor is not None:
                #    self.interactor.stop()
                logger.info('Generating shallow copies of probe, object and parameters and runtime')
                minimal = u.Param()
                minimal.probe = {ID : S._to_dict() for ID,S in self.probe.S.items()}
                minimal.obj = {ID : S._to_dict() for ID,S in self.obj.S.items()}
                minimal.pars = self.p.copy()#_to_dict(Recursive=True)
                minimal.runtime = self.runtime.copy()
                content=minimal
                        
            h5opt = io.h5options['UNSUPPORTED']
            io.h5options['UNSUPPORTED'] = 'ignore'
            logger.info('Saving to %s' % destfile)
            io.h5write(destfile,header=header,content=content)
            io.h5options['UNSUPPORTED'] = h5opt
        else:
            pass
        # we have to wait for all processes, just in case the script isn't finished after saving
        parallel.barrier()
        return destfile
        
    def print_stats(self,table_format = None, detail='summary'):
        """
        Calculates the memrory usage and other info of ptycho instance 
        """
        offset = 8
        active_pods = sum(1 for pod in self.pods.values() if pod.active)
        all_pods = len(self.pods.values())
        info = '\n'
        info += "Process #%d ---- Total Pods %d (%d active) ----" % (parallel.rank,all_pods,active_pods )+'\n'
        info += '-'*80 +'\n'
        desc =dict([('memory','Memory'),('shape','Shape'),('psize','Pixel size'),('dimension','Dimensions'),('views','Views')])
        units = dict([('memory','(MB)'),('shape','(Pixel)'),('psize','(meters)'),('dimension','(meters)'),('views','act.')])
        _table = [('memory',6),('shape',16),('psize',15),('dimension',15),('views',5)]
        table_format = _table if table_format is None else table_format
        h1="(C)ontnr".ljust(offset)
        h2="(S)torgs".ljust(offset)
        for key,column in table_format:
            h1 += " : " + desc[key].ljust(column)
            h2 += " : " + units[key].ljust(column)
        
        info += h1 + '\n' + h2 +'\n'
        info += '-'*80 +'\n'
           
        
        for ID,C in self.containers.iteritems():
            info += C.formatted_report(offset,table_format)
        
        info += '\n'
        if str(detail)!='summary':
            for ID,C in self.containers.iteritems():
                info += C.report()
        

        logger.info(info,extra={'allprocesses':True})
        #logger.debug(info,extra={'allprocesses':True})

    def plot_overview(self,fignum=100):
        """
        plots whole the first four layers of every storage in probe, object % diff
        """
        from matplotlib import pyplot as plt
        plt.ion()
        for s in self.obj.S.values():
            u.plot_storage(s,fignum,'linear',(slice(0,4),slice(None),slice(None)))
            fignum+=1
        for s in self.probe.S.values():
            u.plot_storage(s,fignum,'linear',(slice(0,4),slice(None),slice(None)))
            fignum+=1
        for s in self.diff.S.values():
            u.plot_storage(s,fignum,'log',(slice(0,4),slice(None),slice(None)), cmap='CMRmap')
            fignum+=1
