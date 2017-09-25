# -*- coding: utf-8 -*-
"""
ptycho - definition of the upper-level class Ptycho.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
import time
import paths
import os
import sys

from .. import utils as u
from ..utils.verbose import logger, _, report, headerline, log
from ..utils import parallel
from .. import engines
from ..io import interaction
from classes import Base, Container, Storage, PTYCHO_PREFIX
from manager import ModelManager
from . import model

__all__ = ['Ptycho', 'DEFAULT', 'DEFAULT_io']

DEFAULT_runtime = u.Param(
    run=os.path.split(sys.argv[0])[1].split('.')[0],
    engine="None",
    iteration=0,
    iterations=0,
)

DEFAULT_autoplot = u.Param(
    imfile="plots/%(run)s/%(run)s_%(engine)s_%(iteration)04d.png",
    threaded=True,
    interval=1,
    layout='default',
    dump=True,
    make_movie=False,
)

DEFAULT_autosave = u.Param(
    # If None or False : no saving else save with this given interval
    interval=10,
    # List of probe IDs for autosaving, if None save all [Not implemented]
    probes=None,
    # List of object IDs for autosaving, if None, save all [Not implemented]
    objects=None,
    rfile="dumps/%(run)s/%(run)s_%(engine)s_%(iteration)04d.ptyr",
)

DEFAULT_io = u.Param(
    # Auto save options: (None or False, int). If an integer,
    # specifies autosave interval, if None or False: no autosave
    autosave=DEFAULT_autosave,
    # Plotting parameters for a client
    autoplot=DEFAULT_autoplot,
    # Client-server communication
    interaction=interaction.Server_DEFAULT.copy(),
    home='./',
    rfile="recons/%(run)s/%(run)s_%(engine)s.ptyr"
)
"""Default io parameters. See :py:data:`.io` and a short listing below"""

DEFAULT = u.Param(
    # Verbosity level
    verbose_level=3,
    # Start an ipython kernel for debugging
    ipython_kernel=False,
    # Precision for reconstruction: 'single' or 'double'.
    data_type='single',
    # Do actually nothing if True [not implemented]
    dry_run=False,
    run=None,
    # POD creation rules.
    scan=u.Param(),
    scans=u.Param(),
    # Reconstruction algorithms
    engines={},
    engine=engines.DEFAULTS.copy(),
    io=DEFAULT_io.copy(depth=2)
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
    CType,FType : numpy.dtype
        numpy dtype for arrays. `FType` is for data, i.e. real-valued
        arrays, `CType` is for complex-valued arrays

    interactor : ptypy.io.interaction.Server
        ZeroMQ interaction server for communication with e.g. plotting
        clients

    runtime : Param
        Runtime information, e.g. errors, iteration etc.

    modelm : ModelManager
        THE managing instance for :any:`POD`, :any:`View` and
        :any:`Geo` instances

    probe,obj,exit,diff,mask : Container
        Container instances for illuminations, samples, exit waves,
        diffraction data and detector masks / weights
    """
    DEFAULT = DEFAULT
    """ Default ptycho parameters which is the trunk of the
        default :ref:`ptypy parameter tree <parameters>`"""

    _PREFIX = PTYCHO_PREFIX

    def __init__(self, pars=None, level=2, **kwargs):
        """
        Parameters
        ----------
        pars : Param
            Input parameters, subset of the
            :ref:`ptypy parameter tree<parameters>`

        level : int
            Determines how much is initialized.

            - <= 0 : empty ptypy structure
            - 1 : reads parameters, configures interaction server,
                  see :py:meth:`init_structures`
            - 2 : also configures Containers, initializes ModelManager
                  see :py:meth:`init_data`
            - 3 : also initializes ZeroMQ-communication
                  :py:meth:`init_communication`
            - 4 : also initializes reconstruction engines,
                  see :py:meth:`init_engine`
            - >= 4 : also and starts reconstruction
                    see :py:meth:`run`
        """
        super(Ptycho, self).__init__(None, 'Ptycho')

        # Abort if we load complete structure
        if level <= 0:
            return

        self.p = self.DEFAULT.copy(depth=99)
        """ Reference for parameter tree, with which
            this instance was constructed. """

        # Continue with initialization from parameters
        if pars is not None:
            self.p.update(pars, in_place_depth=3)

        # That may be a little dangerous
        self.p.update(kwargs)

        # Instance attributes

        # Structures
        self.probe = None
        self.obj = None
        self.exit = None
        self.diff = None
        self.mask = None
        self.modelm = None

        # Data
        self.datasource = None

        # Communication
        self.interactor = None
        self.plotter = None

        # Early boot strapping
        self._configure()

        if level >= 1:
            logger.info('\n' + headerline('Ptycho init level 1', 'l'))
            self.init_structures()
        if level >= 2:
            logger.info('\n' + headerline('Ptycho init level 2', 'l'))
            self.init_data()
        if level >= 3:
            logger.info('\n' + headerline('Ptycho init level 3', 'l'))
            self.init_communication()
        if level >= 4:
            logger.info('\n' + headerline('Ptycho init level 4', 'l'))
            self.init_engine()
        if level >= 5:
            self.run()
            self.finalize()

    def _configure(self):
        """
        Early boot strapping.
        """
        p = self.p

        #################################
        # IPython kernel
        #################################
        if parallel.master and p.ipython_kernel:
            u.ipython_kernel.start_ipython_kernel({'Ptycho': self})

        #################################
        # Global logging level
        #################################
        u.verbose.set_level(p.verbose_level)

        #################################
        # Global data type switch
        #################################

        self.data_type = p.data_type
        assert p.data_type in ['single', 'double']
        self.FType = np.dtype(
            'f' + str(np.dtype(np.typeDict[p.data_type]).itemsize)).type
        self.CType = np.dtype(
            'c' + str(2 * np.dtype(np.typeDict[p.data_type]).itemsize)).type
        logger.info(_('Data type', self.data_type))

        # Check if there is already a runtime container
        if not hasattr(self, 'runtime'):
            self.runtime = u.Param()  # DEFAULT_runtime.copy()

        if not hasattr(self, 'engines'):
            # Create an engines entry if it does not already exist
            from collections import OrderedDict
            self.engines = OrderedDict()

        # Generate all the paths
        self.paths = paths.Paths(p.io)

        # Find run name
        self.runtime.run = self.paths.run(p.run)

    def init_communication(self):
        """
        Called on __init__ if ``level >= 3``.

        Initializes ZeroMQ communication on the master node and
        spawns an optional plotting client.
        """
        iaction = self.p.io.interaction
        autoplot = self.p.io.autoplot

        if parallel.master and iaction:
            # Create the interaction server
            self.interactor = interaction.Server(iaction)

            # Register self as an accessible object for the client
            self.interactor.objects['Ptycho'] = self

            # Start the thread
            logger.info('Will start interaction server here: %s:%d'
                        % (self.interactor.address, self.interactor.port))
            port = self.interactor.activate()

            if port is None:
                logger.warn('Interaction server initialization failed. '
                            'Continuing without server.')
                self.interactor = None
                self.plotter = None
            else:
                # Modify port
                iaction.port = port

                # Inform the audience
                log(4, 'Started interaction got the following parameters:'
                    + report(self.interactor.p, noheader=True))

                # Start automated plot client
                self.plotter = None
                if (parallel.master and autoplot and autoplot.threaded and
                        autoplot.interval > 0):
                    from multiprocessing import Process
                    logger.info('Spawning plot client in new Process.')
                    self.plotter = Process(target=u.spawn_MPLClient,
                                           args=(iaction, autoplot,))
                    self.plotter.start()
        else:
            # No interaction wanted
            self.interactor = None
            self.plotter = None

        parallel.barrier()

    def init_structures(self):
        """
        Called on __init__ if ``level >= 1``.

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
        Called on __init__ if ``level >= 2``.

        Creates a datasource and calls for :py:meth:`ModelManager.new_data()`
        Prints statistics on the ptypy structure if ``print_stats=True``
        """
        # Create the data source object, which give diffraction frames one
        # at a time, supporting MPI sharing.
        self.datasource = self.modelm.make_datasource()

        # Load the data. This call creates automatically the scan managers,
        # which create the views and the PODs.
        self.modelm.new_data()

        # Print stats
        parallel.barrier()
        if print_stats:
            self.print_stats()

        # Create plotting instance (maybe)

    def _init_engines(self):
        """
        * deprecated*
        Initialize engines from parameters. Sets :py:attr:`engines`
        """
        # Store the engines in a dict
        self.engines = {}

        # Store the run labels in a list to ensure precedence is preserved.
        self.run_labels = []

        # Loop through p.engines sub-dictionaries
        for run_label, pars in self.p.engines.iteritems():
            # Copy common parameters
            engine_pars = self.p.engine.common.copy()

            # Identify engine by name
            engine_class = engines.by_name(pars.name)

            # Update engine type specific parameters
            engine_pars.update(self.p.engine[pars.name])

            # Update engine instance specific parameters
            engine_pars.update(pars)

            # Create instance
            engine = engine_class(self, engine_pars)

            # Store info
            self.engines[run_label] = engine
            self.run_labels.append(run_label)

    def init_engine(self, label=None, epars=None):
        """
        Called on __init__ if ``level >= 4``.

        Initializes engine with label `label` from parameters and lists
        it internally in ``self.engines`` which is an ordered dictionary.

        Parameters
        ----------
        label : str
            Label of engine which is to be created from parameter set of
            the same label in input parameter tree. If ``None``, an engine
            is created for each available parameter set in input parameter
            tree sorted by label.

        epars : Param or dict
            Set of engine parameters. The created engine is listed as
            *auto00*, *auto01* , etc in ``self.engines``
        """
        if epars is not None:
            # Receiving a parameter set means a new engine parameter set
            # needs to be listed in self.p
            engine_label = 'auto%02d' + len(self.engines)

            # List parameters
            self.p.engines[engine_label] = epars

            # Start over
            self.init_engine(engine_label)

            return engine_label

        elif label is not None:
            try:
                pars = self.p.engines[label]
            except KeyError('No parameter set available for engine label %s\n'
                            'Skipping..' % label):
                pass
            # Copy common parameters
            engine_pars = self.p.engine.common.copy()

            # Identify engine by name
            engine_class = engines.by_name(pars.name)

            # Update engine type specific parameters
            engine_pars.update(self.p.engine[pars.name])

            # Update engine instance specific parameters
            engine_pars.update(pars)

            # Create instance
            engine = engine_class(self, engine_pars)

            # Attach label
            engine.label = label

            # Store info
            self.engines[label] = engine
        else:
            # No label = prepare all engines
            for label in sorted(self.p.engines.keys()):
                self.init_engine(label)

    @property
    def pods(self):
        """ Dict of all :any:`POD` instances in the pool of self """
        return self._pool.get('P', {})

    @property
    def containers(self):
        """ Dict of all :any:`Container` instances in the pool of self """
        return self._pool['C']

    def run(self, label=None, epars=None, engine=None):
        """
        Called on __init__ if ``level >= 5``.

        Start the reconstruction with at least one engine.
        As a consequence, ``self.runtime`` will be filled with content.

        Parameters
        ----------
        label : str, optional
            Engine label of engine to run. If ``None`` all available
            engines are run in the order they were stored in
            ``self.engines``. If the engine is not yet created,
            :py:meth:`init_engine` is called for that label.

        epars : dict or Param, optional
            Engine parameter set. An engine is created from this set,
            using :py:meth:`init_engine` and run immediately afterwards.
            For parameters see :py:data:`.engine`

        engine : BaseEngine, optional
            An engine instance that should be a subclass of
            :py:class:`BaseEngine` or have the same methods.
        """
        if engine is not None:
            # Work with that engine
            if self.runtime.get('start') is None:
                self.runtime.start = time.asctime()

            # Check if there is already a runtime info collector
            if self.runtime.get('iter_info') is None:
                self.runtime.iter_info = []

            # Note when the last autosave was carried out
            if self.runtime.get('last_save') is None:
                self.runtime.last_save = 0

            # Maybe not needed
            if self.runtime.get('last_plot') is None:
                self.runtime.last_plot = 0

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

                # Last minute preparation before a contiguous block of
                # iterations
                engine.prepare()

                auto_save = self.p.io.autosave
                if auto_save is not None and auto_save.interval > 0:
                    if engine.curiter % auto_save.interval == 0:
                        auto = self.paths.auto_file(self.runtime)
                        logger.info(headerline('Autosaving'))
                        self.save_run(auto, 'dump')
                        self.runtime.last_save = engine.curiter
                        logger.info(headerline())

                # One iteration
                engine.iterate()

                # Display runtime information and do saving
                if parallel.master:
                    info = self.runtime.iter_info[-1]
                    # Calculate error:
                    # err = np.array(info['error'].values()).mean(0)
                    err = info['error']
                    logger.info('Iteration #%(iteration)d of %(engine)s :: '
                                'Time %(duration).2f' % info)
                    logger.info('Errors :: Fourier %.2e, Photons %.2e, '
                                'Exit %.2e' % tuple(err))

                parallel.barrier()

            # Done. Let the engine finish up
            engine.finalize()

            # Save
            if self.p.io.rfile:
                self.save_run()
            else:
                pass
            # Time the initialization
            self.runtime.stop = time.asctime()

        elif epars is not None:
            # A fresh set of engine parameters arrived.
            label = self.init_engine(epars=epars)
            self.run(label=label)

        elif label is not None:
            # Looks if there already exists a prepared engine
            # If so, use it, else create one and use it
            engine = self.engines.get(label, None)
            if engine is not None:
                self.run(engine=engine)
            else:
                self.init_engine(label=label)
                self.run(label=label)
        else:
            # Prepare and run ALL engines in self.p.engines
            self.init_engine()
            self.runtime.allstart = time.asctime()
            self.runtime.allstop = None
            for engine in self.engines.values():
                self.run(engine=engine)

    def finalize(self):
        """
        Cleanup
        """
        # 'allstop' will be interpreted as 'quit' on threaded plot clients
        self.runtime.allstop = time.asctime()
        if parallel.master and self.interactor is not None:
            self.interactor.process_requests()
        if self.plotter and self.p.io.autoplot.make_movie:
            logger.info('Waiting for Client to make movie ')
            u.pause(5)
        try:
            # Not so clean.
            self.plotter.join()
        except BaseException:
            pass
        try:
            self.interactor.stop()
        except BaseException:
            pass

    def _run(self, run_label=None):
        """
        *deprecated*
        Start the reconstruction. Former method
        """
        # Time the initialization
        if self.runtime.get('start') is None:
            self.runtime.start = time.asctime()

        # Check if there is already a runtime info collector
        if self.runtime.get('iter_info') is None:
            self.runtime.iter_info = []

        # Note when the last autosave was carried out
        if self.runtime.get('last_save') is None:
            self.runtime.last_save = 0

        # Maybe not needed
        if self.runtime.get('last_plot') is None:
            self.runtime.last_plot = 0

        # Run all engines sequentially
        for run_label in self.run_labels:

            # Set a new engine
            engine = self.engines[run_label]
            # self.current_engine = engine

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

                # Last minute preparation before a contiguous block of
                # iterations
                engine.prepare()

                if self.p.autosave is not None and self.p.autosave.interval > 1:
                    if engine.curiter % self.p.autosave.interval == 0:
                        auto = self.paths.auto_file(self.runtime)
                        logger.info(headerline('Autosaving'), 'l')
                        self.save_run(auto, 'dump')
                        self.runtime.last_save = engine.curiter
                        logger.info(headerline())

                # One iteration
                engine.iterate()

                # Display runtime information and do saving
                if parallel.master:
                    info = self.runtime.iter_info[-1]
                    # Calculate error:
                    err = np.array(info['error'].values()).mean(0)
                    logger.info('Iteration #%(iteration)d of %(engine)s :: '
                                'Time %(duration).2f' % info)
                    logger.info('Errors :: Fourier %.2e, Photons %.2e, '
                                'Exit %.2e' % tuple(err))

                parallel.barrier()
            # Done. Let the engine finish up
            engine.finalize()

            # Save
            # Deactivated for now as something fishy happens through MPI
            self.save_run()

        # Clean up - if needed.

        # Time the initialization
        self.runtime.stop = time.asctime()

    @classmethod
    def _from_dict(cls, dct):
        # This method will be called from save_load on linking
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
            Ptycho instance with ``level == 2``
        """
        import save_load
        from .. import io

        # Determine if this is a .pty file
        # FIXME: do not rely on ".pty" extension.
        if not runfile.endswith('.pty') and not runfile.endswith('.ptyr'):
            logger.warning(
                'Only ptypy file type allowed for continuing a reconstruction')
            logger.warning('Exiting..')
            return None

        logger.info('Creating Ptycho instance from %s' % runfile)
        header = u.Param(io.h5read(runfile, 'header')['header'])
        if header['kind'] == 'minimal':
            logger.info('Found minimal ptypy dump')
            content = u.Param(io.h5read(runfile, 'content')['content'])

            logger.info('Creating new Ptycho instance')
            P = Ptycho(content.pars, level=1)

            logger.info('Attaching probe and object storages')
            for ID, s in content['probe'].items():
                s['owner'] = P.probe
                S = Storage._from_dict(s)
                P.probe._new_ptypy_object(S)
            for ID, s in content['obj'].items():
                s['owner'] = P.obj
                S = Storage._from_dict(s)
                P.obj._new_ptypy_object(S)
                # S.owner=P.obj

            logger.info('Attaching original runtime information')
            P.runtime = content['runtime']
            # P.paths.runtime = P.runtime

        elif header['kind'] == 'fullflat':
            P = save_load.link(io.h5read(runfile, 'content')['content'])

            logger.info('Configuring data types, verbosity '
                        'and server-client communication')

            P._configure()

            logger.info('Reconfiguring sharing rules')  # and loading data')
            print u.verbose.report(P.p)
            P.modelm.sharing_rules = model.parse_model(P.modelm.p['sharing'],
                                                       P.modelm.sharing)

            logger.info('Regenerating exit waves')
            P.exit.reformat()
            P.modelm._initialize_exit(P.pods.values())
            """
            logger.info('Attaching datasource')
            P.datasource = P.modelm.make_datasource(P.p.data)

            logger.info('Reconfiguring sharing rules and loading data')
            P.modelm.sharing_rules = model.parse_model(P.p.model['sharing'],
                                                       P.modelm.sharing)
            P.modelm.new_data()


            """
        if load_data:
            logger.info('Loading data')
            P.init_data()
        return P

    def save_run(self, alt_file=None, kind='minimal', force_overwrite=True):
        """
        Save run to file.

        As for now, diffraction / mask data is not stored

        Parameters
        ----------
        alt_file : str
            Alternative filepath, will override io.save_file

        kind : str
            Type of saving, one of:

                - *'minimal'*, only initial parameters, probe and object
                  storages and runtime information is saved.
                - *'full_flat'*, (almost) complete environment

        """
        import save_load
        from .. import io

        dest_file = None

        if parallel.master:

            if alt_file is not None:
                dest_file = u.clean_path(alt_file)
            else:
                dest_file = self.paths.recon_file(self.runtime)

            header = {'kind': kind,
                      'description': 'Ptypy .h5 compatible storage format'}

            import os
            if os.path.exists(dest_file):
                if force_overwrite:
                    logger.warn('Save file exists but will be overwritten '
                                '(force_overwrite is True)')
                elif not force_overwrite:
                    raise RuntimeError('File %s exists! Operation cancelled.'
                                       % dest_file)
                elif force_overwrite is None:
                    ans = raw_input('File %s exists! Overwrite? [Y]/N'
                                    % dest_file)
                    if ans and ans.upper() != 'Y':
                        raise RuntimeError('Operation cancelled by user.')

            if kind == 'fullflat':
                self.interactor.stop()
                logger.info('Deleting references for interactor, '
                            'datasource and engines.')
                del self.interactor
                del self.datasource
                del self.paths
                try:
                    del self.engines
                    del self.current_engine
                except:
                    pass

                logger.info(
                    'Clearing numpy arrays for exit, diff and mask containers.')

                # self.exit.clear()

                try:
                    for pod in self.pods.values():
                        del pod.exit
                except AttributeError:
                    self.exit.clear()

                self.diff.clear()
                self.mask.clear()
                logger.info('Unlinking and saving to %s' % dest_file)
                content = save_load.unlink(self)
                # io.h5write(dest_file, header=header, content=content)

            elif kind == 'dump':
                # if self.interactor is not None:
                #    self.interactor.stop()
                logger.info('Generating copies of probe, object and parameters '
                            'and runtime')
                dump = u.Param()
                dump.probe = {ID: S._to_dict()
                              for ID, S in self.probe.storages.items()}
                dump.obj = {ID: S._to_dict()
                            for ID, S in self.obj.storages.items()}
                dump.pars = self.p.copy()  # _to_dict(Recursive=True)
                dump.runtime = self.runtime.copy()
                # Discard some bits of runtime to save space
                if len(self.runtime.iter_info) > 0:
                    dump.runtime.iter_info = [self.runtime.iter_info[-1]]

                content = dump

            elif kind == 'minimal':
                # if self.interactor is not None:
                #    self.interactor.stop()
                logger.info('Generating shallow copies of probe, object and '
                            'parameters and runtime')
                minimal = u.Param()
                minimal.probe = {ID: S._to_dict()
                                 for ID, S in self.probe.storages.items()}
                minimal.obj = {ID: S._to_dict()
                               for ID, S in self.obj.storages.items()}
                minimal.pars = self.p.copy()  # _to_dict(Recursive=True)
                minimal.runtime = self.runtime.copy()
                content = minimal

            h5opt = io.h5options['UNSUPPORTED']
            io.h5options['UNSUPPORTED'] = 'ignore'
            logger.info('Saving to %s' % dest_file)
            io.h5write(dest_file, header=header, content=content)
            io.h5options['UNSUPPORTED'] = h5opt
        else:
            pass
        # We have to wait for all processes, just in case the script isn't
        # finished after saving
        parallel.barrier()
        return dest_file

    def print_stats(self, table_format=None, detail='summary'):
        """
        Calculates the memory usage and other info of ptycho instance
        """
        offset = 8
        active_pods = sum(1 for pod in self.pods.values() if pod.active)
        all_pods = len(self.pods.values())
        info = ['\n',
                "Process #%d ---- Total Pods %d (%d active) ----"
                % (parallel.rank, all_pods, active_pods) + '\n',
                '-' * 80 + '\n']

        header = True
        for ID, C in self.containers.iteritems():
            info.append(C.formatted_report(table_format,
                                           offset,
                                           include_header=header))
            if header:
                header = False

        info.append('\n')
        if str(detail) != 'summary':
            for ID, C in self.containers.iteritems():
                info.append(C.report())

        logger.info(''.join(info), extra={'allprocesses': True})
        # logger.debug(info, extra={'allprocesses': True})

    def plot_overview(self, fignum=100):
        """
        plots whole the first four layers of every storage in probe, object
        % diff
        """
        from matplotlib import pyplot as plt
        plt.ion()
        for s in self.obj.storages.values():
            u.plot_storage(s,
                           fignum,
                           'linear',
                           (slice(0, 4), slice(None), slice(None)))
            fignum += 1
        for s in self.probe.storages.values():
            u.plot_storage(s,
                           fignum,
                           'linear',
                           (slice(0, 4), slice(None), slice(None)))
            fignum += 1
        for s in self.diff.storages.values():
            u.plot_storage(s,
                           fignum,
                           'log',
                           (slice(0, 4), slice(None), slice(None)),
                           cmap='CMRmap')
            fignum += 1
        for s in self.mask.storages.values():
            u.plot_storage(s,
                           fignum,
                           'log',
                           (slice(0, 1), slice(None), slice(None)),
                           cmap='gray')
            fignum += 1
