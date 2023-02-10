# -*- coding: utf-8 -*-
"""
ptycho - definition of the upper-level class Ptycho.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
import time
import json
from . import paths
from collections import OrderedDict

from .. import utils as u
from ..utils.verbose import logger, _, report, headerline, log, LogTime
from ..utils.verbose import ilog_message, ilog_streamer, ilog_newline
from ..utils import parallel
from .. import engines
from .classes import Base, Container, Storage, PTYCHO_PREFIX
from .manager import ModelManager
from .. import defaults_tree

# This needs to be done here as it populates the defaults tree
from .. import __has_zmq__
if __has_zmq__:
    from ..io import interaction

__all__ = ['Ptycho']



@defaults_tree.parse_doc('ptycho')
class Ptycho(Base):
    """
    Ptycho : A ptychographic data holder and reconstruction manager.

    This is the highest level class. It organizes and contains the data,
    manages the reconstruction engines, and interacts with the outside world.

    If MPI is enabled, this class acts both as a manager (rank = 0) and
    a worker (any rank), and most information exists on all processes.
    In its original design, the only part that is divided between processes is the
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

    ~Ptycho.model : ModelManager
        THE managing instance for :any:`POD`, :any:`View` and
        :any:`Geo` instances

    ~Ptycho.probe, ~Ptycho.obj,~Ptycho.exit,~Ptycho.diff,~Ptycho.mask : Container
        Container instances for illuminations, samples, exit waves,
        diffraction data and detector masks / weights


    Defaults:

    [verbose_level]
    default = 'ERROR'
    help = Verbosity level
    doc = Verbosity level for information logging.
       - ``CRITICAL``: Only critical errors
       - ``ERROR``:    All errors
       - ``WARNING``:  Warning
       - ``INFO``:     Process Information
       - ``INSPECT``:  Object Information
       - ``DEBUG``:    Debug
    type = str, int
    userlevel = 0

    [data_type]
    default = 'single'
    help = Reconstruction floating number precision
    doc = Reconstruction floating number precision (``'single'`` or
          ``'double'``)
    type = str
    userlevel = 1

    [run]
    default = None
    help = Reconstruction identifier
    doc = Reconstruction run identifier. If ``None``, the run name will
          be constructed at run time from other information.
    type = str
    userlevel = 0

    [frames_per_block]
    default = 100000
    help = Max number of frames per block of data
    doc = This parameter determines the size of buffer arrays for GPUs.
          Reduce this number if you run out of memory on the GPU.
    type = int
    lowlim = 1
    userlevel = 1

    [dry_run]
    default = False
    help = Dry run switch
    doc = Run everything skipping all memory and cpu-heavy steps (and
          file saving). **NOT IMPLEMENTED**
    type = bool
    userlevel = 2

    [ipython_kernel]
    default = False
    type = bool
    help = Start an ipython kernel for debugging
    doc = Start an ipython kernel for debugging.

    [io]
    default = None
    type = Param
    help = Global parameters for I/O
    doc = Global parameter container for I/O settings.

    [io.home]
    default = "./"
    type = str
    help = Base directory for all I/O
    doc = home is the root directory for all input/output operations. All other path parameters that
      are relative paths will be relative to this directory.

    [io.rfile]
    default = "recons/%(run)s/%(run)s_%(engine)s_%(iterations)04d.ptyr"
    type = str
    help = Reconstruction file name (or format string)
    doc = Reconstruction file name or format string (constructed against runtime dictionary)

    [io.rformat]
    default = "minimal"
    type = str
    help = Reconstruction file format
    doc = Choose a reconstruction file format for after engine completion.
       - ``'minimal'``: Bare minimum of information
       - ``'dls'``:    Custom format for Diamond Light Source
    choices = 'minimal','dls'

    [io.interaction]
    default = None
    type = Param
    help = ZeroMQ interactor options
    doc = Options for the communications server

    [io.interaction.active]
    default = True
    type = bool
    help = turns on the interaction
    doc = If True the interaction starts, if False all interaction is turned off

    [io.interaction.server]
    default =
    type = @io.interaction.server
    help = Link to server parameter tree

    [io.interaction.client]
    default =
    type = @io.interaction.client
    help = Link to client parameter tree

    [io.autosave]
    default = Param
    type = Param
    help = Auto-save options
    doc = Options for automatic saving during reconstruction.

    [io.autosave.active]
    default = True
    type = bool
    help = Activation switch
    doc = If ``True`` the current reconstruction will be saved at regular intervals. 

    [io.autosave.interval]
    default = 10
    type = int
    help = Auto-save interval
    doc = If ``>0`` the current reconstruction will be saved at regular intervals according to the
    pattern in :py:data:`paths.autosave` . If ``<=0`` not automatic saving
    lowlim = -1

    [io.autosave.rfile]
    default = "dumps/%(run)s/%(run)s_%(engine)s_%(iterations)04d.ptyr"
    type = str
    help = Auto-save file name (or format string)
    doc = Auto-save file name or format string (constructed against runtime dictionary)

    [io.autoplot]
    default = Param
    type = Param
    help = Plotting client parameters
    doc = Container for the plotting.

    [io.autoplot.active]
    default = True
    type = bool
    help = Activation switch
    doc = If ``True`` the current reconstruction will be plotted at regular intervals. 

    [io.autoplot.imfile]
    default = "plots/%(run)s/%(run)s_%(engine)s_%(iterations)04d.png"
    type = str
    help = Plot images file name (or format string)
    doc = Plot images file name (or format string).
    userlevel = 1

    [io.autoplot.interval]
    default = 1
    type = int
    help = Number of iterations between plot updates
    doc = Requests to the server will happen with this iteration intervals. Note that this will work
      only if interaction.polling_interval is smaller or equal to this number. If ``interval
      =0`` plotting is disabled which should be used, when ptypy is run on a cluster.
    lowlim = -1

    [io.autoplot.threaded]
    default = True
    type = bool
    help = Live plotting switch
    doc = If ``True``, a plotting client will be spawned in a new thread and connected at
      initialization. If ``False``, the master node will carry out the plotting, pausing the
      reconstruction. This option should be set to ``True`` when ptypy is run on an isolated
      workstation.

    [io.autoplot.layout]
    default = "default"
    type = str
    help = Options for default plotter or template name
    doc = Flexible layout for default plotter is not implemented yet. Please choose one of the
      templates ``'default'``,``'black_and_white'``,``'nearfield'``, ``'minimal'`` or ``'weak'``
    userlevel = 2

    [io.autoplot.dump]
    default = False
    type = bool
    help = Switch to dump plots as image files
    doc = Switch to dump plots as image files during reconstruction.

    [io.autoplot.make_movie]
    default = False
    type = bool
    help = Produce reconstruction movie after the reconstruction.
    doc = Switch to request the production of a movie from the dumped plots at the end of the
      reconstruction.

    [io.benchmark]
    default = None
    type = str
    help = Produce timings for benchmarking the performance of data loaders and engines
    doc = Switch to get timings and save results to a json file in p.io.home
        Choose ``'all'`` for timing data loading, engine_init, engine_prepare, engine_iterate and engine_finalize
    userlevel = 2

    [scans]
    default = None
    type = Param
    help = Container for instances of scan parameters
    doc =

    [scans.*]
    default =
    type = @scan.*
    help = Wildcard entry for list of scans to load. See :py:data:`scan`

    [engines]
    default = None
    type = Param
    help = Container for instances of engine parameters
    doc =

    [engines.*]
    default =
    type = @engine.*
    help = Wildcard entry for list of engines to run. See :py:data:`engine`
    doc = The value of engines.*.name is used to choose among the available engines.

    """

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
                  see :py:meth:`init_communication`
            - 4 : also initializes reconstruction engines,
                  see :py:meth:`init_engine`
            - >= 4 : also and starts reconstruction
                    see :py:meth:`run`
        """
        super(Ptycho, self).__init__(None, 'Ptycho')

        # Create a parameter structure from the the class-level defaults
        self.p = self.DEFAULT.copy(99)

        # Abort if we load complete structure
        if level <= 0:
            return

        # Continue with initialization from parameters
        if pars is not None:
            self.p.update(pars, in_place_depth=99)

        # That may be a little dangerous
        self.p.update(kwargs)

        # Validate the incoming parameters
        # FIXME : Validation should maybe happen for each class that uses the
        #         the parameters, i.e. like a depth=1 validation
        defaults_tree['ptycho'].validate(self.p)
        # Instance attributes

        # Structures
        self.probe = None
        self.obj = None
        self.exit = None
        self.diff = None
        self.mask = None
        self.model = None
        self.new_data = None

        # Communication
        self.interactor = None
        self.plotter = None
        self.record_positions = False
        self._jupyter_client = None

        # Early boot strapping
        self._configure()

        # Keep a bibliography
        self.citations = u.Bibliography()
        self.citations.add_article(
            title='A computational framework for ptychographic reconstructions',
            author='Enders B. and Thibault P.',
            journal='Proc. Royal Soc. A',
            volume=472,
            year=2016,
            page=20160640,
            doi='10.1098/rspa.2016.0640',
            comment='The Ptypy framework',
        )

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
            'f' + str(np.dtype(np.sctypeDict[p.data_type]).itemsize)).type
        self.CType = np.dtype(
            'c' + str(2 * np.dtype(np.sctypeDict[p.data_type]).itemsize)).type
        logger.info(_('Data type', self.data_type))
        # Check if there is already a runtime container
        if not hasattr(self, 'runtime'):
            self.runtime = u.Param()  # DEFAULT_runtime.copy()

        if not hasattr(self, 'scans'):
            # Create a scans entry if it does not already exist
            self.scans = OrderedDict()

        if not hasattr(self, 'engines'):
            # Create an engines entry if it does not already exist
            self.engines = OrderedDict()

        # Generate all the paths
        self.paths = paths.Paths(p.io)

        # Todo: determine block size based on memory available
        self.frames_per_block = p.frames_per_block

        # Find run name
        self.runtime.run = self.paths.run(p.run)

        # Benchmark
        if self.p.io.benchmark == 'all':
            self.benchmark = u.Param()
            self.benchmark.data_load = 0
            self.benchmark.engine_init = 0
            self.benchmark.engine_prepare = 0
            self.benchmark.engine_iterate = 0
            self.benchmark.engine_finalize = 0

    def init_communication(self):
        """
        Called on __init__ if ``level >= 3``.

        Initializes ZeroMQ communication on the master node and
        spawns an optional plotting client.
        """
        iaction = self.p.io.interaction
        autoplot = self.p.io.autoplot

        if __has_zmq__ and parallel.master and iaction.active:
            # Create the interaction server
            self.interactor = interaction.Server(iaction.server)

            # Register self as an accessible object for the client
            self.interactor.objects['Ptycho'] = self

            # Start the thread
            logger.info('Will start interaction server here: %s:%d'
                        % (self.interactor.address, self.interactor.port))
            port = self.interactor.activate()

            if port is None:
                logger.warning('Interaction server initialization failed. '
                            'Continuing without server.')
                self.interactor = None
                self.plotter = None
            else:
                # Modify port
                iaction.server.port = port

                # Inform the audience
                log(4, 'Started interaction got the following parameters:'
                    + report(self.interactor.p, noheader=True))

                # Start automated plot client
                self.plotter = None

                if (parallel.master and autoplot.active and autoplot.threaded and
                        autoplot.interval > 0):
                    from multiprocessing import Process
                    logger.info('Spawning plot client in new Process.')
                    self.plotter = Process(target=u.spawn_MPLClient,
                                           args=(iaction.client, autoplot,))
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
        :py:attr:`model` and the containers :py:attr:`probe` for
        illumination, :py:attr:`obj` for the samples, :py:attr:`exit` for
        the exit waves, :py:attr:`diff` for diffraction data and
        :py:attr:`Ptycho.mask` for detectors masks
        """
        self.probe = Container(self, ID='Cprobe', data_type='complex')
        self.obj = Container(self, ID='Cobj', data_type='complex')
        self.exit = Container(self, ID='Cexit', data_type='complex', distribution="scattered")
        self.diff = Container(self, ID='Cdiff', data_type='real', distribution="scattered")
        self.mask = Container(self, ID='Cmask', data_type='bool', distribution="scattered")
        # Initialize the model manager. This also initializes the
        # containers.
        self.model = ModelManager(self, self.p.scans)
    
    def init_data(self, print_stats=True):
        """
        Called on __init__ if ``level >= 2``.

        Call :py:meth:`ModelManager.new_data()`
        Prints statistics on the ptypy structure if ``print_stats=True``
        """
        # Load the data. This call creates automatically the scan managers,
        # which create the views and the PODs. Sets self.new_data
        with LogTime(self.p.io.benchmark == 'all') as t:
            self.new_data = self.model.new_data()
        if (self.p.io.benchmark == 'all') and parallel.master: self.benchmark.data_load += t.duration

        # Print stats
        parallel.barrier()
        if print_stats:
            self.print_stats()

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
            engine_label = 'auto%02d' % len(self.engines)

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

            # Identify engine by name
            engine_class = engines.by_name(pars.name)

            # Create instance
            engine = engine_class(self, pars)

            # Attach label
            engine.label = label

            # Store info
            self.engines[label] = engine
        else:
            # No label = prepare all engines
            for label in sorted(self.p.engines.keys()):
                # FIXME workaround to avoid parameter trees that are just meant as templates.
                try:
                    self.p.engines[label].name
                except:
                    continue
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

        engine : ~ptypy.engines.base.BaseEngine, optional
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
            ilog_message('%s: initializing engine' %engine.p.name)
            with LogTime(self.p.io.benchmark == 'all') as t:
                engine.initialize()
            if (self.p.io.benchmark == 'all') and parallel.master: self.benchmark.engine_init += t.duration

            # One .prepare() is always executed, as Ptycho may hold data
            ilog_message('%s: preparing engine' %engine.p.name)
            self.new_data = [(d.label, d) for d in self.diff.S.values()]
            with LogTime(self.p.io.benchmark == 'all') as t:
                engine.prepare()
            if (self.p.io.benchmark == 'all') and parallel.master: self.benchmark.engine_prepare += t.duration

            # Start the iteration loop
            ilog_streamer('%s: starting engine' %engine.p.name)
            while not engine.finished:
                # Check for client requests
                if parallel.master and self.interactor is not None:
                    self.interactor.process_requests()

                parallel.barrier()

                # Check for new data
                with LogTime(self.p.io.benchmark == 'all') as t:
                    self.new_data = self.model.new_data()
                if (self.p.io.benchmark == 'all') and parallel.master: self.benchmark.data_load += t.duration

                # Last minute preparation before a contiguous block of
                # iterations
                if self.new_data:
                    with LogTime(self.p.io.benchmark == 'all') as t:
                        engine.prepare()
                    if (self.p.io.benchmark == 'all') and parallel.master: self.benchmark.engine_prepare += t.duration

                auto_save = self.p.io.autosave
                if auto_save.active and auto_save.interval > 0:
                    if engine.curiter % auto_save.interval == 0:
                        auto = self.paths.auto_file(self.runtime)
                        logger.info(headerline('Autosaving'))
                        self.save_run(auto, 'dump')
                        self.runtime.last_save = engine.curiter
                        logger.info(headerline())

                # One iteration
                with LogTime(self.p.io.benchmark == 'all') as t:
                    engine.iterate()
                if (self.p.io.benchmark == 'all') and parallel.master: self.benchmark.engine_iterate += t.duration

                # Display runtime information and do saving
                if parallel.master:
                    info = self.runtime.iter_info[-1]
                    # Calculate error:
                    # err = np.array(info['error'].values()).mean(0)
                    err = info['error']
                    logger.info('Iteration #%(iteration)d of %(engine)s :: '
                                'Time %(duration).3f' % info)
                    logger.info('Errors :: Fourier %.2e, Photons %.2e, '
                                'Exit %.2e' % tuple(err))
                    imsg = '%(engine)s: Iteration # %(iteration)d/%(numiter)d :: ' %info + \
                                   'Fourier %.2e, Photons %.2e, Exit %.2e' %tuple(err)
                    if not self.p.io.autoplot.threaded:
                        if not (info["iteration"] % self.p.io.autoplot.interval):
                            if self._jupyter_client is None:
                                from IPython import display
                                from ptypy.utils.plot_client import _JupyterClient
                                self._jupyter_client = _JupyterClient(self, autoplot_pars=self.p.io.autoplot, layout_pars=self.p.io.autoplot.layout)
                            self._jupyter_client.runtime.update(self.runtime)
                            display.display(self._jupyter_client.plot(title=imsg), clear=True)
                    else:
                        ilog_streamer(imsg)
                    
                parallel.barrier()

            ilog_newline()

            # Done. Let the engine finish up
            with LogTime(self.p.io.benchmark == 'all') as t:
                engine.finalize()
            if (self.p.io.benchmark == 'all') and parallel.master: self.benchmark.engine_finalize += t.duration

            # Save
            if self.p.io.rfile:
                self.save_run(kind=self.p.io.rformat)
            else:
                pass
            # Time the initialization
            self.runtime.stop = time.asctime()

            # Save benchmarks to json file
            if (self.p.io.benchmark == 'all') and parallel.master:
                try:
                    with open(self.paths.home + "/benchmark.json", "w") as json_file:
                        json.dump(self.benchmark, json_file)
                    logger.info("Benchmarks have been written to %s" %self.paths.home + "/benchmark.json")
                except Exception as e:
                    logger.warning("Failed to write benchmarks to file: %s" %e)

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
            if not self.engines: self.init_engine()
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

        # Hint at citations (for all log levels)
        citation_info = '\n'.join([headerline('This reconstruction relied on the following work', 'l', '='),
        str(self.citations),
        headerline('', 'l', '=')])
        log("CITATION", citation_info)

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
        from . import save_load
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
        if header['kind'] == 'minimal' or header['kind'] == 'dls':
            logger.info('Found minimal ptypy dump')
            content = io.h5read(runfile, 'content')['content']

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

            logger.info('Regenerating exit waves')
            P.exit.reformat()
            P.model._initialize_exit(list(P.pods.values()))

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
                  storages, positions and runtime information is saved.
                - *'full_flat'*, (almost) complete environment

        """
        from . import save_load
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
                    logger.warning('Save file exists but will be overwritten '
                                '(force_overwrite is True)')
                elif not force_overwrite:
                    raise RuntimeError('File %s exists! Operation cancelled.'
                                       % dest_file)
                elif force_overwrite is None:
                    ans = input('File %s exists! Overwrite? [Y]/N'
                                    % dest_file)
                    if ans and ans.upper() != 'Y':
                        raise RuntimeError('Operation cancelled by user.')

            if kind == 'fullflat':
                self.interactor.stop()
                logger.info('Deleting references for interactor '
                            ' and engines.')
                del self.interactor
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
                for ID, S in self.probe.storages.items():
                    dump.probe[ID]['grids'] = S.grids()

                dump.obj = {ID: S._to_dict()
                            for ID, S in self.obj.storages.items()}

                for ID, S in self.obj.storages.items():
                    dump.obj[ID]['grids'] = S.grids()

                try:
                    defaults_tree['ptycho'].validate(self.p) # check the parameters are actually able to be read back in
                except RuntimeError:
                    logger.warning("The parameters we are saving won't pass a validator check!")
                dump.pars = self.p.copy()  # _to_dict(Recursive=True)
                dump.runtime = self.runtime.copy()
                # Discard some bits of runtime to save space
                if len(self.runtime.iter_info) > 0:
                    dump.runtime.iter_info = [self.runtime.iter_info[-1]]

                content = dump

            elif kind == 'minimal' or kind == 'dls':
                # if self.interactor is not None:
                #    self.interactor.stop()
                logger.info('Generating shallow copies of probe, object and '
                            'parameters and runtime')
                minimal = u.Param()
                minimal.probe = {ID: S._to_dict()
                                 for ID, S in self.probe.storages.items()}

                minimal.obj = {ID: S._to_dict()
                               for ID, S in self.obj.storages.items()}
                try:
                    defaults_tree['ptycho'].validate(self.p) # check the parameters are actually able to be read back in
                except RuntimeError:
                    logger.warning("The parameters we are saving won't pass a validator check!")
                minimal.pars = self.p.copy()  # _to_dict(Recursive=True)
                minimal.runtime = self.runtime.copy()

                content = minimal
            else:
                raise RuntimeError("Save file format '" + str(kind) + "' is not supported")

            if kind == 'dls':
                for ID, S in self.probe.storages.items():
                    content.probe[ID]['grids'] = S.grids()

                for ID, S in self.obj.storages.items():
                    content.obj[ID]['grids'] = S.grids()

            if kind in ['minimal', 'dls'] and self.record_positions:
                content.positions = {}
                for ID, S in self.obj.storages.items():
                    content.positions[ID] = np.array([v.coord for v in S.views if v.pod.pr_view.layer==0])

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
        for ID, C in self.containers.items():
            info.append(C.formatted_report(table_format,
                                           offset,
                                           include_header=header))
            if header:
                header = False

        info.append('\n')
        if str(detail) != 'summary':
            for ID, C in self.containers.items():
                info.append(C.report())

        if parallel.size <= 5:
            logger.info(''.join(info), extra={'allprocesses': True})
        else:
            logger.info(''.join(info))
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

    
    def _redistribute_data(self, div = 'rect', obj_storage=None):
        """
        This function redistributes data among nodes, so that each
        node becomes in charge of a contiguous block of scanning
        positions.
        Each node is associated with a domain of the scanning pattern,
        and communication happens node-to-node after each has worked
        out which of its pods are not part of its domain.
        """
        t0 = time.time()
        
        if obj_storage is None:
            for s in self.obj.storages.values():
                out = self._redistribute_data(div=div, obj_storage=s)
            return out
        else:
            # Not effective for object modes, but these are rare anyway
            views = obj_storage.views
            # get the range of positions 
            pos = np.array([v.coord for v in views])
            mn = pos.min(0)
            mx = pos.max(0)
            diff = mx - mn
            # expand the outer borders slightly to avoid edge effects
            mn -= 0.01 * diff
            mx += 0.01 * diff
            # in [0,1]
            normed = lambda x : (x-mn) / (mx-mn)
                
            N = parallel.size
            if div == 'rect':
                roots = np.arange(int(np.sqrt(N)),0,-1)
                i = roots[N % roots == 0][0]
                assert (i * (N / i) == N)
                
                def __node(pos):
                    x,y = normed(pos)
                    rank = int( i * x) + i * int(N/i * y )
                    return rank
                
            elif div == 'angle':
                def __node(pos):
                    x,y = normed(pos) -.5
                    return  int(min(N * (np.angle(x+ 1.j * y) / np.pi+1.)/2.,N-1))
                    
            elif div == 'row':
                __node = lambda pos : int(normed(pos)[0] * N)
                
            elif div == 'column':
                __node = lambda pos : int(normed(pos)[1] * N)
    
            # Now get the the views to the diffraction data
            views =  dict([(v.ID,v.pod.di_view) for v in views])
       
            # Determine which views need to be moved.            
            # This relies heavily on the per-node tagging of active/passive
            destinations = {}
            sources = {}
            positions = []
            label = []
            for name,view in views.items():
                pos = view.pod.ob_view.coord
                _dest = __node(pos) % N
                label.append(_dest)
                positions.append(normed(pos))
                if _dest == parallel.rank:
                    # Nothing to do
                    continue
                else:
                    destinations[name] = _dest
                if view.active:
                    sources[name] = parallel.rank
            destinations = parallel.gather_dict(destinations)
            destinations = parallel.bcast_dict(destinations)
            sources = parallel.gather_dict(sources)
            sources = parallel.bcast_dict(sources)
            if not destinations:
                logger.info('No data redistribution necessary.')
                return positions, label
            
            ## Pairing
            pairs = {}
            # prepare (enlarge) the storages on the receiving nodes
            for name, source in sources.items():
                dest = destinations[name] # This must work
                pairs[name] = (source,dest)
                view = views[name]
                if dest == parallel.rank:
                    # receiving this pod, so mark it as active
                    view.active = True
                    view.pod.ma_view.active = True
                    view.pod.ex_view.active = True
            for name in ['Cdiff', 'Cmask']:
                self.containers[name].reformat()
    
            # transfer data
            transferred = 0
            for name, (source,dest) in pairs.items():
                view = views[name]
                if parallel.rank == source:
                    parallel.send(view.data, dest=dest)
                    parallel.send(view.pod.mask, dest=dest)
                    view.active = False
                    view.pod.ma_view.active = False
                    view.pod.ex_view.active = False
                    transferred += 1
                if dest == parallel.rank:
                    # your turn to receive
                    view.data = parallel.receive()
                    view.pod.mask = parallel.receive()
                parallel.barrier()
                
            for name in ['Cdiff', 'Cmask', 'Cexit']:
                self.containers[name].reformat()
                
            transferred = parallel.comm.reduce(transferred)
            t1 = time.time()
    
            if parallel.master:
                logger.info('Redistributed data, moved %u pods in %.2f s'
                            %  (transferred, t1 - t0))
    
        return positions, label

    def _best_decomposition(self, N):
        """
        Work out the best arrangement of domains for a given number of
        nodes. Assumes a roughly square scan.
        """
