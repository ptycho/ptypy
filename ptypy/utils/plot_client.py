"""
Client tools for plotting.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import time
import numpy as np
from threading import Thread, Lock
import os

from .verbose import logger, report, log
from .parameters import Param
from .array_utils import crop_pad, clean_path
from .plot_utils import plt # pyplot import with all the specialized settings
from .plot_utils import PtyAxis, imsave, pause, rmphaseramp

__all__ = ['MPLClient', 'MPLplotter', 'PlotClient', 'spawn_MPLClient', 'TEMPLATES', 'DEFAULT',
           'figure_from_ptycho', 'figure_from_ptyr']


Storage_DEFAULT = Param(
    # maybe we would want this container specific
    clims=[None, [-np.pi, np.pi]],
    cmaps=['gray', 'hsv'],
    crop=[0.3, 0.3],  # fraction of array to crop for display
    rm_pr=True,  # remove_phase_ramp = True
    shape=None,  # if None the shape is determining
    auto_display=['a', 'p'],  # quantities to display
    layers=None,  # (int or list or None)
    local_error=False,  # plot a local error map (ignored in probe)
    use_colorbar = True,
    mask = 0.3, # Fraction (radius) of data to use for clims (if None) or phase_ramp removal
)

DEFAULT = Param()
DEFAULT.figsize = (16, 10)
DEFAULT.dpi = 100
DEFAULT.ob = Storage_DEFAULT.copy()
DEFAULT.pr = Storage_DEFAULT.copy()
DEFAULT.pr.auto_display = ['c']
DEFAULT.simplified_aspect_ratios = False
DEFAULT.gridspecpars = (0.1, 0.12, 0.07, 0.95, 0.05, 0.93)
DEFAULT.plot_error = [True, True, True]  # decide which error to plot
DEFAULT.interactive = True
DEFAULT.home = '/tmp/ptypy/'
DEFAULT.movie = False

TEMPLATES = Param({'default':DEFAULT})
bnw = DEFAULT.copy(depth=4)
bnw.ob.cmaps=['gray','bone']
bnw.pr.cmaps=['gray','bone']
bnw.pr.clims=[None, None]
bnw.ob.clims=[None, None]
bnw.pr.auto_display=['a','p']
TEMPLATES['black_and_white'] = bnw

weak = DEFAULT.copy(depth=4)
weak.ob.cmaps=['gray','bone']
weak.ob.clims=[None, None]
TEMPLATES['weak'] = weak

minimal = bnw.copy(depth=4)
minimal.ob.cmaps=['gray','jet']
minimal.ob.layers=[0]
minimal.pr.cmaps=['gray','jet']
minimal.pr.layers=[0]
minimal.simplified_aspect_ratios = True
TEMPLATES['minimal'] = minimal

nearfield = DEFAULT.copy(depth=4)
nearfield.pr.clims=[None, [-np.pi, np.pi]]
nearfield.pr.cmaps=['gray', 'hsv']
nearfield.pr.crop=[0.0, 0.0]  # fraction of array to crop for display
nearfield.pr.rm_pr=False  #
nearfield.ob.clims=[None, None]
nearfield.ob.cmaps=['gray', 'jet']
nearfield.ob.crop=[0.0, 0.0]  # fraction of array to crop for display
nearfield.ob.rm_pr=False #
TEMPLATES['nearfield'] = nearfield

nfbnw = DEFAULT.copy(depth=4)
nfbnw.pr.clims=[None, [-np.pi, np.pi]]
nfbnw.pr.cmaps=['gray', 'hsv']
nfbnw.pr.crop=[0.0, 0.0]  # fraction of array to crop for display
nfbnw.pr.rm_pr=False  #
nfbnw.ob.clims=[None, None]
nfbnw.ob.cmaps=['gray', 'bone']
nfbnw.ob.crop=[0.0, 0.0]  # fraction of array to crop for display
nfbnw.ob.rm_pr=False #
TEMPLATES['nf_black_and_white'] = nfbnw

jupyter = DEFAULT.copy(depth=4)
jupyter.pr.clims=[None, [-np.pi, np.pi]]
jupyter.pr.cmaps=['gray','hsv']
jupyter.pr.crop=[0.0, 0.0]
jupyter.pr.auto_display = ['c']
jupyter.rm_pr=False
jupyter.ob.clims=[None, None]
jupyter.ob.cmaps=['gray','viridis']
jupyter.ob.crop=[0.2, 0.2]
jupyter.ob.mask = 0.5
jupyter.rm_pr=True
jupyter.figsize=(16,8)
jupyter.dpi = 60
jupyter.simplified_aspect_ratios = True
jupyter.gridspecpars = (0.2, 0.12, 0.07, 0.95, 0.05, 0.93)
TEMPLATES['jupyter'] = jupyter

del nfbnw
del nearfield
del bnw
del weak
del minimal

class PlotClient(object):
    """
    A client that connects and continually gets the data required for plotting.
    Note: all data is transferred as soon as the server provides it. This might
    be a waste of bandwidth if all that is required is a client that plots "on demand"...

    This PlotClient doesn't actually plot.
    """

    ACTIVE = 1
    DATA = 2
    STOPPED = 0

    def __init__(self, client_pars=None, in_thread=False):
        """
        Create a client and attempt to connect to a running reconstruction server.
        """
        # This avoids circular imports.
        from ptypy.io.interaction import Client

        self.log_level = 5 if in_thread else 3
        # If client_pars is None, connect with the defaults defined in ptypy.io.interaction
        self.client = Client(client_pars)

        # Data requests are stored in this dictionary:
        # self.cmd_dct['cmd'] = [ticket, buffer, key]
        # When the data associated with a ticket arrives it is places in buffer[key].
        self.cmd_dct = {}

        # Initialize data containers. Here we use our own "Param" class, which adds attribute access
        # on top of dictionary.
        self.pr = Param()  # Probe
        self.ob = Param()  # Object
        self.runtime = Param()  # Runtime information (contains reconstruction metrics)
        self._new_data = False

        # The thread that will manage incoming data in the background
        self._thread = None
        self._stopping = False
        self._lock = Lock()
        self._has_stopped = False

        # Here you should initialize plotting capabilities.
        self.config = None

        # A small flag
        self.is_initialized = False

    @property
    def status(self):
        """
        True only if new data were acquired since last time checked.
        """
        if self._new_data:
            self._new_data = False
            return self.DATA
        elif self._has_stopped:
            return self.STOPPED
        else:
            return self.ACTIVE

    def start(self):
        """
        This needs to be run for the thread to initialize.
        """
        self._stopping = False
        self._thread = Thread(target=self._loop)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        """
        Stop loop plot
        """
        self._stopping = True
        # Gedenkminute
        pause(0.1)
        self.disconnect()
        self._thread.join(0.2)

    def get_data(self):
        """
        Thread-safe way to copy data buffer.
        :return:
        """
        with self._lock:
            pr = self.pr.copy(depth=10)
            ob = self.ob.copy(depth=10)
            if self.runtime.get('last_info') is not None:
                self.runtime['iter_info'].append(self.runtime['last_info'].copy())
            else:
                from ptypy.engines import DEFAULT_iter_info
                self.runtime['iter_info'].append(DEFAULT_iter_info)
            runtime = self.runtime.copy(depth=10)
        return pr, ob, runtime

    def _connect(self):
        """
        Connect to the reconstruction server.
        """
        self.client.activate()
        log(self.log_level,'Client connecting to server...')

        # Pause until connected
        while not self.client.connected:
            time.sleep(0.1)
        log(self.log_level,'Client connected.')

    def disconnect(self):
        """
        Disconnect from reconstruction server.
        """
        self.client.stop()

    def _initialize(self):
        """
        Wait for reconstruction to start, then grab synchronously basic information
        for initialization.
        """
        log(self.log_level,'Client requesting configuration parameters')
        autoplot = self.client.get_now("Ptycho.p.io.autoplot")
        if hasattr(autoplot,'items'):
            self.config = Param(autoplot)
            self.config.home = self.client.get_now("Ptycho.p.io.home")
        else:
            self.config = autoplot
        log(self.log_level,'Client received the following configuration:')
        log(self.log_level,report(self.config))

        # Wait until reconstruction starts.
        log(self.log_level,'Client waiting for reconstruction to start...')
        ready = self.client.get_now("'iter_info' in Ptycho.runtime")

        # requesting fulll runtime
        log(self.log_level,'Client requesting runtime container')
        self.runtime = Param(self.client.get_now("Ptycho.runtime"))

        try:
            _a = self.runtime['iter_info']
        except KeyError:
            # we've initialied the plotclient before the engine init loop, should create an iter_info list to match.
            # avoids a race condition in engine.init
            self.runtime['iter_info'] = []

        while not ready:
            time.sleep(.1)
            ready = self.client.get_now("'start' in Ptycho.runtime")

        log(self.log_level,'Client ready')

        # Get the list of object IDs
        ob_IDs = self.client.get_now("list(Ptycho.obj.S.keys())")
        log(self.log_level,'1 object to plot.' if len(ob_IDs) == 1 else '%d objects to plot.' % len(ob_IDs))

        # Prepare the data requests
        for ID in ob_IDs:
            S = Param()
            self.ob[ID] = S
            self.cmd_dct["Ptycho.obj.S['%s'].data" % str(ID)] = [None, S, 'data']
            self.cmd_dct["Ptycho.obj.S['%s'].psize" % str(ID)] = [None, S, 'psize']
            self.cmd_dct["Ptycho.obj.S['%s'].center" % str(ID)] = [None, S, 'center']

        # Get the list of probe IDs
        pr_IDs = self.client.get_now("list(Ptycho.probe.S.keys())")
        log(self.log_level,'1 probe to plot.' if len(pr_IDs) == 1 else '%d probes to plot.' % len(pr_IDs))

        # Prepare the data requests
        for ID in pr_IDs:
            S = Param()
            self.pr[ID] = S
            self.cmd_dct["Ptycho.probe.S['%s'].data" % str(ID)] = [None, S, 'data']
            self.cmd_dct["Ptycho.probe.S['%s'].psize" % str(ID)] = [None, S, 'psize']
            self.cmd_dct["Ptycho.probe.S['%s'].center" % str(ID)] = [None, S, 'center']

        # Data request for the error.
        self.cmd_dct["Ptycho.runtime['iter_info'][-1]"] = [None, self.runtime, 'last_info']

        # Get the dump file path
        self.cmd_dct["Ptycho.paths.plot_file(Ptycho.runtime)"] = [None, self.runtime, 'plot_file']

        # Get info if it's all over
        self.cmd_dct["Ptycho.runtime.get('allstop') is not None"] = [None, self.__dict__, '_stopping']


    def _request_data(self):
        """
        Request all data to the server (asynchronous).
        """
        for cmd, item in self.cmd_dct.items():
            item[0] = self.client.get(cmd)

    def _store_data(self):
        """
        Transfer all data from the client to local attributes.
        """
        with self._lock:
            for cmd, item in self.cmd_dct.items():
                item[1][item[2]] = self.client.data[item[0]]
            # An extra step for the error. This should be handled differently at some point.
            # self.error = np.array([info['error'].sum(0) for info in self.runtime.iter_info])
            self._new_data = True
        self.client.flush()

    def _loop(self):
        # Activate the client thread.
        self._connect()

        self._initialize()
        while not self._stopping:
            self._request_data()
            self.client.wait()
            self._store_data()
            # logger.info('New data arrived.')
        self.disconnect()
        self._has_stopped = True

class MPLplotter(object):
    """
    Plotting Client for Ptypy, using matplotlib.
    """
    DEFAULT = DEFAULT

    def __init__(self, pars=None, probes = None, objects= None, runtime= None, in_thread=False):
        """
        Create a client and attempt to connect to a running reconstruction server.
        """
        self.log_level = 5 if in_thread else 3
        # Initialize data containers
        self.pr = probes
        self.ob = objects
        if runtime is None:
            self.runtime = Param()
            self.runtime.iter_info = []
        else:
            self.runtime = runtime
        self._set_autolayout(pars)
        self.pr_plot=Param()
        self.ob_plot=Param()

    def _set_autolayout(self,pars):
        self.p = self.DEFAULT.copy(depth=4)
        plt.interactive(self.p.interactive)
        if pars is not None:
            if str(pars)==pars:
                try:
                    pars = TEMPLATES[pars]
                except KeyError:
                    log(self.log_level,'Plotting template "\\%s" not found, using default settings' % str(pars))

            if hasattr(pars,'items'):
                self.p.update(pars,in_place_depth=4)

    def update_plot_layout(self):
        """
        Generate the plot layout.
        """
        def simplify_aspect_ratios(sh):
            ratio = sh[1] / float(sh[0])
            rp = 1 - int(ratio < 2./3.) + int(ratio >= 3./2.)
            if rp == 0:
                sh = (4, 2)
            elif rp == 2:
                sh = (2, 4)
            else:
                sh = (3, 3)
            return sh
        self.num_shape_list = []
        num_shape_list = self.num_shape_list
        for key in sorted(self.ob.keys()):
            cont = self.ob[key]
            plot = self.p.ob.copy()
            # determine the shape
            sh = cont.data.shape[-2:]
            if self.p.simplified_aspect_ratios:
                sh = simplify_aspect_ratios(sh)
            if plot.use_colorbar:
                sh=(sh[0],int(sh[1]*1.15))

            layers = plot.layers
            if layers is None:
                layers = cont.data.shape[0]
            if np.isscalar(layers):
                layers = list(range(layers))
            plot.layers = layers
            plot.axes_index = len(num_shape_list)
            num_shape = [len(layers)*len(plot.auto_display)+int(plot.local_error), sh]
            num_shape_list.append(num_shape)
            self.ob_plot[key]=plot

        # per default we will use the a frame similar to the last object frame for plotting
        if np.array(self.p.plot_error).any():
            self.error_axes_index = len(num_shape_list)-1
            self.error_frame = num_shape_list[-1][0]
            num_shape_list[-1][0] += 1  # add a frame

        for key in sorted(self.pr.keys()):
            cont = self.pr[key]
            # attach a plotting dict from above
            # this will need tweaking
            plot = self.p.pr.copy()
            # determine the shape
            sh = cont.data.shape[-2:]
            if self.p.simplified_aspect_ratios:
                sh = simplify_aspect_ratios(sh)
            if plot.use_colorbar:
                sh=(sh[0],int(sh[1]*1.2))

            layers = plot.layers
            if layers is None:
                layers = cont.data.shape[0]
            if np.isscalar(layers):
                layers = list(range(layers))
            plot.layers = layers
            plot.axes_index = len(num_shape_list)
            num_shape = [len(layers)*len(plot.auto_display), sh]
            num_shape_list.append(num_shape)
            self.pr_plot[key]=plot

        axes_list, plot_fig, gs = self.create_plot_from_tile_list(1, num_shape_list, self.p.figsize, self.p.dpi)
        sy, sx = gs.get_geometry()
        w, h, l, r, b, t = self.p.gridspecpars
        gs.update(wspace=w*sy, hspace=h*sx, left=l, right=r, bottom=b, top=t)
        self.draw()
        for axes in axes_list:
            for pl in axes:
                plt.setp(pl.get_xticklabels(), fontsize=8)
                plt.setp(pl.get_yticklabels(), fontsize=8)
        self.plot_fig = plot_fig
        self.axes_list = axes_list
        self.gs = gs

    @staticmethod
    def create_plot_from_tile_list(fignum=1, num_shape_list=[(4, (2, 2))], figsize=(8, 8),dpi=100):
        def fill_with_tiles(size, sh, num, figratio=16./9.):
            coords_tl = []
            while num > 0:
                Horizontal = True
                N_h = size[1]//sh[1]
                N_v = size[0]//sh[0]
                # looking for tight fit
                if num <= N_v and np.abs(N_h-num) >= np.abs(N_v-num):
                    Horizontal = False
                elif num <= N_h and np.abs(N_h-num) <= np.abs(N_v-num):
                    Horizontal = True
                elif size[0] == 0 or size[1]/float(size[0]+0.00001) > figratio:
                    Horizontal = True
                else:
                    Horizontal = False

                if Horizontal:
                    N = N_h
                    a = size[1] % sh[1]
                    coords = [(size[0], int(ii*sh[1])+a) for ii in range(N)]
                    size[0] += sh[0]
                else:
                    N = N_v
                    a = size[0] % sh[0]
                    coords = [(int(ii*sh[0])+a, size[1]) for ii in range(N)]
                    size[1] += sh[1]

                num -= N
                coords_tl += coords
                coords_tl.sort()

            return coords_tl, size


        allcoords = []
        fig_aspect_ratio = figsize[0]/float(figsize[1])
        size = [0, 0]
        # determine frame thickness
        aa = np.array([sh[0]*sh[1] for N, sh in num_shape_list])
        N, bigsh = num_shape_list[np.argmax(aa)]
        frame = int(0.2*min(bigsh))
        M,last_sh = num_shape_list[0]
        M=0
        for ii,(N, sh) in enumerate(num_shape_list):
            if not np.allclose(np.asarray(sh),np.asarray(last_sh)) or ii==(len(num_shape_list)-1):
                nsh = np.array(last_sh)+frame
                coords, size = fill_with_tiles(size, nsh, M+N, fig_aspect_ratio)
                M=0
                last_sh = sh
                allcoords+=coords
            else:
                M+=N
                continue

        coords_list =[]
        M=0
        for ii,(N,sh) in enumerate(num_shape_list):
            coords_list.append(allcoords[M:M+N])
            M+=N

        from matplotlib import gridspec
        gs = gridspec.GridSpec(size[0], size[1])
        fig = plt.figure(fignum, dpi=dpi)
        fig.clf()

        mag = min(figsize[0]/float(size[1]), figsize[1]/float(size[0]))
        figsize = (size[1]*mag, size[0]*mag)
        fig.set_size_inches(figsize, forward=True)
        space = 0.1*size[0]
        gs.update(wspace=0.1*size[0], hspace=0.12*size[0], left=0.07, right=0.95, bottom=0.05, top=0.93) #this is still a stupid hardwired parameter
        axes_list=[]
        for (N, sh), coords in zip(num_shape_list, coords_list):
            axes_list.append([fig.add_subplot(gs[co[0]+frame//2:co[0]+frame//2+sh[0], co[1]+frame//2:co[1]+frame//2+sh[1]]) for co in coords])

        return axes_list, fig, gs

    def draw(self):
        if self.p.interactive:
            plt.draw()
            time.sleep(0.1)
        else:
            plt.show()

    def plot_error(self):
        if np.array(self.p.plot_error).any():
            try:
                axis = self.axes_list[self.error_axes_index][self.error_frame]
                # get runtime info
                error = np.array([info['error'] for info in self.runtime.iter_info])
                err_fmag = error[:, 0]
                err_phot = error[:, 1]
                err_exit = error[:, 2]
                axis.clear()
                fmag = err_fmag/np.max(err_fmag) if np.max(err_fmag) > 0 else err_fmag
                axis.plot(fmag, label='err_fmag %2.2f%% of %.2e' % (fmag[-1]*100, np.max(err_fmag)))
                phot = err_phot/np.max(err_phot) if np.max(err_phot) > 0 else err_phot
                axis.plot(phot, label='err_phot %2.2f%% of %.2e' % (phot[-1]*100, np.max(err_phot)))
                ex = err_exit/np.max(err_exit) if np.max(err_exit) > 0 else err_exit
                axis.plot(ex, label='err_exit %2.2f%% of %.2e' % (ex[-1]*100, np.max(err_exit)))
                axis.legend(loc=1, fontsize=10) #('err_fmag %.2e' % np.max(err_fmag),'err_phot %.2e' % np.max(err_phot),'err_exit %.2e' % np.max(err_exit)),
                plt.setp(axis.get_xticklabels(), fontsize=10)
                plt.setp(axis.get_yticklabels(), fontsize=10)
            except:
                pass

    def plot_storage(self, storage, plot_pars, title="", typ='obj'):
        # get plotting paramters
        pp = plot_pars
        axes = self.axes_list[pp.axes_index]
        weight = pp.get('weight')
        # plotting mask for ramp removal
        sh = storage.data.shape[-2:]
        if np.isscalar(pp.mask):
            x, y = np.indices(sh)-np.reshape(np.array(sh)//2, (len(sh),)+len(sh)*(1,))
            mask = (np.sqrt(x**2+y**2) < pp.mask*min(sh)/2.)

        # cropping
        crop = np.array(sh)*np.array(pp.crop)//2
        data = storage.data
        #crop = -crop.astype(int)
        #data = crop_pad(storage.data, crop, axes=[-2, -1])
        #plot_mask = crop_pad(mask, crop, axes=[-2, -1])

        pty_axes = pp.get('pty_axes',[])
        for layer in pp.layers:
            for ind,channel in enumerate(pp.auto_display):
                ii = layer*len(pp.auto_display)+ind
                if ii >= len(axes):
                    break
                try:
                    ptya = pty_axes[ii]
                except IndexError:
                    cmap = pp.cmaps[ind % len(pp.cmaps)] #if ind[1]=='p' else pp.cmaps[0]
                    ptya = PtyAxis(axes[ii], data = data[layer], channel=channel,cmap = cmap)
                    ptya.set_mask(mask, False)
                    if pp.clims is not None and pp.clims[ind] is not None:
                        ptya.set_clims(pp.clims[ind][0],pp.clims[ind][1], False)
                    if pp.use_colorbar:
                        ptya.add_colorbar()
                    ptya._resize_cid = self.plot_fig.canvas.mpl_connect('resize_event', ptya._after_resize_event)
                    pty_axes.append(ptya)
                # get the layer
                ptya.set_mask(mask, False)
                ptya.set_data(data[layer])
                ptya.ax.set_ylim(crop[0],sh[0]-crop[0])
                ptya.ax.set_xlim(crop[1],sh[1]-crop[1])
                #ptya._update_colorbar()
                if channel == 'c':
                    if typ == 'obj':
                        mm = np.mean(np.abs(data[layer]*mask)**2)
                        info = 'T=%.2f' % mm
                    else:
                        mm = np.sum(np.abs(data[layer])**2)
                        info = 'P=%1.1e' % mm
                    ttl = '%s#%d (C)\n%s' % (title, layer, info)
                elif channel == 'a':
                    ttl = '%s#%d (a)' % (title, layer)
                else:
                    ttl = '%s#%d (p)' % (title, layer)
                ptya.ax.set_title(ttl, size=12)

        pp.pty_axes = pty_axes

    def save(self, pattern, count=0):
        try:
            r = self.runtime.copy(depth=1)
            r.update(r.iter_info[-1])
            plot_file = clean_path(pattern % r)
        except BaseException:
            log(self.log_level,'Could not auto generate image dump file from runtime.')
            plot_file = 'ptypy_%05d.png' % count

        log(self.log_level,'Dumping plot to %s' % plot_file)
        self.plot_fig.savefig(plot_file,dpi=300)
        folder,fname = os.path.split(plot_file)
        mode ='w' if count==1 else 'a'
        self._framefile = folder+os.path.sep+'frames.txt'
        with open(self._framefile,mode) as f:
            f.write(plot_file+'\n')
            f.close()

    def plot_all(self):
        for key, storage in self.pr.items():
            #print key
            pp = self.pr_plot[key]
            self.plot_storage(storage,pp, str(key), 'pr')
        for key, storage in self.ob.items():
            #print key
            pp = self.ob_plot[key]
            self.plot_storage(storage,pp, str(key), 'obj')
        self.plot_error()
        self.draw()

class MPLClient(MPLplotter):

    DEFAULT = DEFAULT

    def __init__(self, client_pars=None, autoplot_pars=None, home=None,\
                 layout_pars=None, in_thread=False, is_slave=False):
        
        from ptypy.core.ptycho import Ptycho
        self.config = Ptycho.DEFAULT.io.autoplot.copy(depth=3)
        self.config.update(autoplot_pars)
        # set a home directory
        self.config.home = home if home is not None else self.DEFAULT.get('home')

        layout = self.config.get('layout',layout_pars)

        super(MPLClient,self).__init__(pars = layout, in_thread = in_thread)

        self.pc = PlotClient(client_pars,in_thread=in_thread)
        self.pc.start()
        self._framefile= None
        self.is_slave = is_slave

    def loop_plot(self):
        """
        Plot forever.
        """
        count = 0
        initialized = False
        while True:
            status = self.pc.status
            if status == self.pc.DATA:
                self.pr, self.ob, runtime = self.pc.get_data()
                self.runtime.update(runtime)
                if not initialized:
                    if self.is_slave and self.pc.config:
                        self.config.update(self.pc.config)

                    self._set_autolayout(self.config.layout)
                    self.update_plot_layout()
                    initialized=True

                self.plot_all()
                self.draw()
                count+=1
                if self.config.dump:
                    self.save(self.config.home + self.config.imfile,count)
                    #plot_file = clean_path(runtime['plot_file'])

            elif status == self.pc.STOPPED:
                break
            pause(.1)

        if self.config.get('make_movie'):

            from ptypy import utils as u
            u.png2mpg(self._framefile, RemoveImages=True)

class _JupyterClient(MPLplotter):

    DEFAULT = DEFAULT

    def __init__(self, ptycho, autoplot_pars=None, layout_pars=None):
        from ptypy.core.ptycho import Ptycho
        self.config = Ptycho.DEFAULT.io.autoplot.copy(depth=3)
        self.config.update(autoplot_pars)
        layout = self.config.get('layout',layout_pars)

        super(_JupyterClient,self).__init__(pars=layout, 
                                      objects=ptycho.obj.S,
                                      probes=ptycho.probe.S,
                                      runtime=ptycho.runtime,
                                      in_thread=False)
        self.initialized = False

    def plot(self, title=""):
        if not self.initialized:
            self.update_plot_layout()
            self.initialized=True
        self.plot_fig.suptitle(title)
        self.plot_all()
        plt.close(self.plot_fig)
        return self.plot_fig

def figure_from_ptycho(P, pars=None):
    """
    Returns a matplotlib figure displaying a reconstruction
    from a Ptycho instance.

    Parameters
    ----------
    P : Ptycho
        Ptycho instance
    pars : Plotting paramters, optional
        plotting template as u.Param() istance

    Return
    ------
    fig : matplotlib.figure.Figure
    """
    if pars is None:
        pars = TEMPLATES["jupyter"]
    plotter = MPLplotter(pars=pars, 
                         objects=P.obj.S,
                         probes=P.probe.S,
                         runtime=P.runtime,
                         in_thread=False)
    plotter.update_plot_layout()
    plotter.plot_all()
    return plotter.plot_fig

def figure_from_ptyr(filename, pars=None):
    """
    Returns a matplotlib figure displaying a reconstruction
    from a .ptyr file.

    Parameters
    ----------
    filename : str
        path to .ptyr file
    pars : Plotting paramters, optional
        plotting template as u.Param() istance

    Return
    ------
    fig : matplotlib.figure.Figure
    """
    from ..io import h5read
    header = h5read(filename,'header')['header']
    if str(header['kind']) == 'fullflat':
        raise NotImplementedError('Loading specific data from flattened dump not yet supported')
    else: 
        content = list(h5read(filename,'content').values())[0]
        runtime = content['runtime']
        probes = Param()
        probes.update(content['probe'], Convert = True)
        objects = Param()
        objects.update(content['obj'], Convert = True)
    if pars is None:
        pars = TEMPLATES["jupyter"]
    plotter = MPLplotter(pars=pars, 
                         objects=objects,
                         probes=probes,
                         runtime=runtime,
                         in_thread=False)
    plotter.update_plot_layout()
    plotter.plot_all()
    return plotter.plot_fig

class Bragg3dClient(object):
    """
    MPLClient analog for 3d Bragg data, which needs to be reduced to 2d
    before plotting.
    """

    def __init__(self, client_pars=None, autoplot_pars=None, home=None,
                 in_thread=False, is_slave=False):

        from ptypy.core.ptycho import Ptycho
        self.p = Ptycho.DEFAULT.io.autoplot.copy(depth=3)
        self.p.update(autoplot_pars)
        # need a home directory
        self.p.home = home if home is not None else DEFAULT.get('home')

        self.runtime = Param()
        self.ob = Param()
        self.pr = Param()

        self.log_level = 5 if in_thread else 3

        self.pc = PlotClient(client_pars, in_thread=in_thread)
        self.pc.start()

        # set up axes
        self.plotted = False
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.ion()
        fig, self.ax = plt.subplots(nrows=2, ncols=2)
        self.plot_fig = fig
        self.ax_err = self.ax[1,1]
        self.ax_obj = (self.ax[0,0], self.ax[0,1], self.ax[1,0])

    def loop_plot(self):
        """
        Plot forever.
        """
        count = 0
        initialized = False
        while True:
            status = self.pc.status
            if status == self.pc.DATA:
                self.pr, self.ob, runtime = self.pc.get_data()
                self.runtime.update(runtime)
                self.plot_all()
                count+=1
                if self.p.dump:
                    self.save(self.p.home + self.p.imfile, count)
                    #plot_file = clean_path(runtime['plot_file'])

            elif status == self.pc.STOPPED:
                break
            self.plt.pause(.1)

        if self.p.get('make_movie'):
            from ptypy import utils as u
            u.png2mpg(self._framefile, RemoveImages=True)

    def plot_all(self):
        self.plot_error()
        self.plot_object()
        self.plot_probe()

        if 'shrinkwrap' in self.runtime.iter_info[-1].keys():
            self.plot_shrinkwrap()

    def plot_shrinkwrap(self):
        try:
            self.ax_shrinkwrap
        except:
            _, self.ax_shrinkwrap = plt.subplots()
        sx, sprofile, low, high = self.runtime.iter_info[-1]['shrinkwrap']
        iteration = self.runtime.iter_info[-1]['iteration']
        self.ax_shrinkwrap.clear()
        self.ax_shrinkwrap.plot(sx, sprofile, 'b')
        self.ax_shrinkwrap.axvline(low, color='red')
        self.ax_shrinkwrap.axvline(high, color='red')
        self.ax_shrinkwrap.set_title('iter: %d, interval: %.3e'
            %(iteration, (high-low)))

    def plot_object(self):

        data = list(self.ob.values())[0]['data'][0]
        center = list(self.ob.values())[0]['center']
        psize = list(self.ob.values())[0]['psize']
        lims_r3 = (-center[0] * psize[0], (data.shape[0] - center[0]) * psize[0])
        lims_r1 = (-center[1] * psize[1], (data.shape[1] - center[1]) * psize[1])
        lims_r2 = (-center[2] * psize[2], (data.shape[2] - center[2]) * psize[2])

        if self.plotted:
            for ax_ in self.ax_obj:
                ax_.old_xlim = ax_.get_xlim()
                ax_.old_ylim = ax_.get_ylim()
                ax_.clear()

        arr = np.mean(np.abs(data), axis=2).T # (r1, r3) from top left
        arr = np.flipud(arr)                  # (r1, r3) from bottom left
        self.ax_obj[0].imshow(arr, interpolation='none',
            extent=lims_r3+lims_r1) # extent changes limits, not image orientation
        self.plt.setp(self.ax_obj[0], ylabel='r1', xlabel='r3', title='side view')

        arr = np.mean(np.abs(data), axis=1).T # (r2, r3) from top left
        self.ax_obj[1].imshow(arr, interpolation='none',
            extent=lims_r3+lims_r2[::-1])
        self.plt.setp(self.ax_obj[1], ylabel='r2', xlabel='r3', title='top view')

        arr = np.mean(np.abs(data), axis=0)   # (r1, r2) from top left
        arr = np.flipud(arr)                  # (r1, r2) from bottom left
        self.ax_obj[2].imshow(arr, interpolation='none',
            extent=lims_r2+lims_r1)
        self.plt.setp(self.ax_obj[2], ylabel='r1', xlabel='r2', title='front view')

        for ax_ in self.ax_obj:
            ax_.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

        if self.plotted:
            for ax_ in self.ax_obj:
                ax_.set_xlim(ax_.old_xlim)
                ax_.set_ylim(ax_.old_ylim)

        self.plotted = True

    def plot_probe(self):
        pass

    def plot_error(self):
        # error
        error = np.array([info['error'] for info in self.runtime.iter_info])
        err_fmag = error[:, 0] / np.max(error[:, 0])
        err_phot = error[:, 1] / np.max(error[:, 1])
        err_exit = error[:, 2] / np.max(error[:, 2])

        self.ax_err.clear()
        self.ax_err.plot(err_fmag, label='err_fmag')
        self.ax_err.plot(err_phot, label='err_phot')
        self.ax_err.plot(err_exit, label='err_exit')
        self.ax_err.legend(loc='upper right')

    def save(self, pattern, count=0):
        try:
            r = self.runtime.copy(depth=1)
            r.update(r.iter_info[-1])
            plot_file = clean_path(pattern % r)
        except BaseException:
            log(self.log_level,'Could not auto generate image dump file from runtime.')
            plot_file = 'ptypy_%05d.png' % count

        log(self.log_level,'Dumping plot to %s' % plot_file)
        self.plot_fig.savefig(plot_file,dpi=300)
        folder,fname = os.path.split(plot_file)
        mode ='w' if count==1 else 'a'
        self._framefile = folder+os.path.sep+'frames.txt'
        with open(self._framefile,mode) as f:
            f.write(plot_file+'\n')
            f.close()


def spawn_MPLClient(client_pars, autoplot_pars, home=None):
    """
    A function that creates and runs a silent instance of MPLClient.
    """
    cls = MPLClient
    # 3d Bragg is handled by a MPLClient subclass and is identified by the layout
    try:
        if autoplot_pars.layout == 'bragg3d':
            cls = Bragg3dClient
    except:
        pass

    mplc = cls(client_pars, autoplot_pars, home, in_thread=True, is_slave=True)
    try:
        mplc.loop_plot()
    except KeyboardInterrupt:
        pass
    finally:
        print('Stopping plot client...')
        mplc.pc.stop()

if __name__ =='__main__':
    from ptypy.resources import moon_pr, flower_obj
    moon = moon_pr(256)
    flower = flower_obj(512)
    dmoon = np.resize(moon,(2,256,256))
    dflower = np.resize(flower,(2,512,512))
    pr = Param({'Moon':Param({'data':dmoon, 'shape':dmoon.shape,'psize':1})})
    ob = Param({'Flower':Param({'data':dflower, 'shape':dflower.shape,'psize':1})})
    runtime = Param()
    runtime.iter_info = []
    err1= np.array([1e3,1e4,1e5])
    for i in range(10):
        err1 *= np.array([0.95,0.9,0.85])
        error_dict={'v1':err1.copy(),'v2':err1.copy()}
        runtime.iter_info.append(error_dict)

    plotter = MPLplotter(probes = pr,objects=ob, runtime=runtime)
    plotter.update_plot_layout()
    plotter.plot_all()
    plt.draw()
    plt.show()
    pause(10)
