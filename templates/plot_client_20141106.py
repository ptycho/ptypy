"""
A Rough plotting client for Ptypy reconstructions.
"""

# change plotting backend to Qt4
import matplotlib as mpl
mpl.rcParams['backend']='Qt4Agg'
from matplotlib import gridspec
import time
import numpy as np

from ptypy import utils as u
from ptypy.io.interaction import Client, network_DEFAULT


Container_DEFAULT = u.Param(
    # maybe we would want this container specific
    clims = [None,[-np.pi,np.pi]],
    cmaps = ['gray','hsv'],
    crop = [0.4,0.4], # fraction of array to crop for display
    rm_pr = True,  # remove_phase_ramp = True
    shape = None, # if None the shape is determining
    auto_display = ['a','p'], # quantities to display
    layers = None, # (int or list or None)
    local_error = False  # plot a local error map (ignored in probe)
)

DEFAULT = u.Param()
DEFAULT.figsize = (14,10)
DEFAULT.ob=Container_DEFAULT.copy()
DEFAULT.pr=Container_DEFAULT.copy()
DEFAULT.pr.auto_display = ['c']
DEFAULT.simplified_aspect_ratios=False
DEFAULT.gridspecpars=(0.1,0.12,0.07,0.95,0.05,0.93)
DEFAULT.plot_error = [True,True,True]  # decide which error to plot

templates= u.Param()


class Plot_Client(object):
    """
    Plotting Client for Ptypy
    Very simple for now. Does not inherit from Client but 
    holds a client instance.
    TODO:
    - maybe inherit from Client?
    """
    DEFAULT=DEFAULT
    
    def __init__(self, client_pars=None, plot_template=None, interactive=True, **kwargs):
        """
        Create a client and attempt to connect to a running reconstruction server.
        """

        self.client=Client(client_pars)
        self.connect()

        # Initialize data containers
        self.pr=u.Param()
        self.ob=u.Param()
        self.err=[]
        self.ferr=None

        # Initialize the plotter
        from matplotlib import pyplot
        self.interactive = interactive
        self.pp = pyplot
        pyplot.interactive(interactive)
        
        self.templates = templates
        self.p = u.Param(DEFAULT)
        #self.update_plot_layout(plot_template=plot_template)

        # save as 'cmd': tuple(ticket,buffer,key)
        # will call get cmds with 'cmd', save the ticket in <ticket> and
        # save the resulting data in buffer[key]
        
        self.cmd_dct = {}
        self.server_dcts={}
        
    def connect(self):
        """
        Connect to the reconstruction server.
        """
        self.client.activate()

        # pause until connected
        while not self.client.connected:
            time.sleep(0.1)
            
    def disconnect(self):
        """
        Disconnect from reconstruction server.
        """
        self.client.stop()
        
    
    def get_data(self,cmd_list,flush=True):
        """
        Request data from server (waiting for data to be processed).
        """

        c=self.client
        tlist=[]

        # get tickets for requests
        for cmd in cmd_list:
            ticket=c.get(cmd)
            tlist+=[ticket]

        # wait for the last ticket to be processed
        completed = c.wait(ticket)

        # change data for tickets
        data=[c.data[ticket] for ticket in tlist]

        # empty buffer
        if flush: c.flush()

        return data

    def initialize(self):
        """
        Wait for reconstruction to start, then grab synchronously basic information
        for initialization.
        """

        # Wait until reconstruction starts.
        ready = self.client.get_now("'start' in Ptycho.runtime")
        while not ready:
            time.sleep(.1)
            ready = self.client.get_now("'start' in Ptycho.runtime")

        # Build local containers
        self.ob = u.Param()
        self.pr = u.Param()
        self.runtime = u.Param()

        ob_IDs = self.client.get_now("Ptycho.obj.S.keys()")
        for ID in ob_IDs:
            S = u.Param()
            self.ob[ID] = S
            self.cmd_dct["Ptycho.obj.S['%s'].data" % str(ID)] = [None, S, 'data']
            self.cmd_dct["Ptycho.obj.S['%s'].psize" % str(ID)] = [None, S, 'psize']
            self.cmd_dct["Ptycho.obj.S['%s'].center" % str(ID)] = [None, S, 'center']

        pr_IDs = self.client.get_now("Ptycho.probe.S.keys()")
        for ID in pr_IDs:
            S = u.Param()
            self.pr[ID] = S
            self.cmd_dct["Ptycho.probe.S['%s'].data" % str(ID)] = [None, S, 'data']
            self.cmd_dct["Ptycho.probe.S['%s'].psize" % str(ID)] = [None, S, 'psize']
            self.cmd_dct["Ptycho.probe.S['%s'].center" % str(ID)] = [None, S, 'center']

        self.cmd_dct["Ptycho.runtime['iter_info']"] = [None, self.runtime, 'iter_info']

    def request_data(self):
        """
        Request all data to the server (asynchronous).
        """
        for cmd,item in self.cmd_dct.iteritems():
            #print cmd,item[0]
            item[0]=self.client.get(cmd) 
            
    def mv_data_from_client_buffer(self):
        """
        Transfer all data from the client to local attributes.
        """
        for cmd,item in self.cmd_dct.iteritems():
            #print cmd,item
            item[1][item[2]] = self.client.data[item[0]]
            
        self.client.flush()

    def update_plot_layout(self,plot_template=None,**kwargs):
        # generate the plot frame
        def simplify_aspect_ratios(sh):
            ratio= sh[1] / float(sh[0])    
            rp = 1 - int(ratio < 2./3.) + int(ratio >= 3./2.)
            if rp==0:
                sh =(4,2)
            elif rp==2:
                sh =(2,4)
            else:
                sh =(3,3)
            return sh
        # local references:
        ob=self.ob
        pr=self.pr
        ptemplate = self.p
        if plot_template is not None:
            ptemplate = plot_template
        elif ptemplate is None:
            ptemplate = 'legacy'
            
        if ptemplate is not None:
            if isinstance(ptemplate,str):
                template = self.templates.get(ptemplate)
                if template is None:
                    raise RuntimeError('Plot template not known. Look in class.templates.keys() for a template of parameters')
            elif isinstance(ptemplate,dict):
                template=ptemplate
                self.templates.update({'custom':ptemplate})
                
            self.p.update(template)
        
        #axes =[]
        self.num_shape_list=[]
        num_shape_list=self.num_shape_list
        for key in sorted(ob.keys()):
            cont=ob[key]
            #print key
            # attach a plotting dict from above
            # this will need tweaking
            cp=self.p.ob.copy()
            # determine the shape
            sh = cont.data.shape[-2:]
            if self.p.simplified_aspect_ratios:
                sh = simplify_aspect_ratios(sh)
                
            layers=cp.layers
            if layers is None:
                layers=cont.data.shape[0]
            if np.isscalar(layers):
                layers=range(layers)
            cp.layers=layers
            cp.axes_index=len(num_shape_list)
            num_shape=[len(layers)*len(cp.auto_display)+int(cp.local_error),sh]
            num_shape_list.append(num_shape)
            cont.plot = cp
            
        # per default we will use the a frame similar to the last object frame for plotting
        if np.array(self.p.plot_error).any():
            self.error_axes_index = len(num_shape_list)-1
            self.error_frame=num_shape_list[-1][0]
            num_shape_list[-1][0]+=1 # add a frame
        
        for key in sorted(pr.keys()):
            cont=pr[key]
            # attach a plotting dict from above
            # this will need tweaking
            cont.plot=self.p.pr.copy()
            # determine the shape
            sh = cont.data.shape[-2:]
            if self.p.simplified_aspect_ratios:
                sh = simplify_aspect_ratios(sh)
                
            layers=cont.plot.layers
            if layers is None:
                layers=cont.data.shape[0]
            if np.isscalar(layers):
                layers=range(layers)
            cont.plot.layers=layers
            cont.plot.axes_index=len(num_shape_list)
            num_shape=[len(layers)*len(cont.plot.auto_display),sh]
            num_shape_list.append(num_shape)
        
        axes_list,plot_fig,gs = self.create_plot_from_tile_list(1,num_shape_list,self.p.figsize)
        #plot_fig.suptitle(p.paramdict.get('scans')[0])
        #obj_axes = plot_axes_list[0]
        #pr_axes = plot_axes_list[1]
        #err_axes = [pr_axes.pop(0)]
        #local_error_axes=[obj_axes.pop(0) for i in range(lerr_axes_num)]
        sy,sx = gs.get_geometry()
        w,h,l,r,b,t = self.p.gridspecpars
        gs.update(wspace=w*sy,hspace=h*sx,left=l,right=r,bottom=b,top=t)
        self.draw()
        #plot_axes = obj_axes+pr_axes+err_axes
        plot_fig.hold(False)
        for axes in axes_list:
            for pl in axes: 
                pl.hold(False)
                self.pp.setp(pl.get_xticklabels(), fontsize=8) #doesn't do nothin
                self.pp.setp(pl.get_yticklabels(), fontsize=8)
        self.plot_fig = plot_fig
        self.axes_list = axes_list
        self.gs= gs

    def create_plot_from_tile_list(self,fignum=1,num_shape_list=[(4,(2,2))],figsize=(8,8)):        
        def fill_with_tiles(size,sh,num,figratio=16./9.):
            coords_tl=[]
            while num > 0:
                Horizontal = True
                N_h = size[1]//sh[1]
                N_v = size[0]//sh[0]
                #looking for tight fit
                if num<=N_v and np.abs(N_h-num) >= np.abs(N_v-num):
                    Horizontal = False
                elif num<=N_h and np.abs(N_h-num) <= np.abs(N_v-num):
                    Horizontal = True
                elif size[0]==0 or size[1]/float(size[0]+0.00001) > figratio:
                    Horizontal = True
                else:
                    Horizontal = False
                     
                if Horizontal:
                    N=N_h
                    a=size[1]%sh[1]
                    coords=[(size[0],int(ii*sh[1])+a) for ii in range(N)]
                    size[0]+=sh[0]
                else:
                    N=N_v
                    a=size[0]%sh[0]
                    coords=[(int(ii*sh[0])+a,size[1]) for ii in range(N)]
                    size[1]+=sh[1]
                    
                num -=N
                coords_tl+=coords
                coords_tl.sort()
                
            return coords_tl, size
        
        coords_list=[]
        fig_aspect_ratio = figsize[0]/float(figsize[1])
        size=[0,0] 
        # determine frame thickness
        aa= np.array([sh[0]*sh[1] for N,sh in num_shape_list])
        N,bigsh = num_shape_list[np.argmax(aa)]
        frame=int(0.2*min(bigsh))
        for N,sh in num_shape_list:
            nsh=np.array(sh)+frame
            coords, size  = fill_with_tiles(size,nsh,N,fig_aspect_ratio)
            coords_list.append(coords)

        gs=gridspec.GridSpec(size[0],size[1])
        fig = self.pp.figure(fignum)
        fig.clf()
        
        mag=min(figsize[0]/float(size[1]),figsize[1]/float(size[0]))
        figsize=(size[1]*mag,size[0]*mag)
        fig.set_size_inches(figsize,forward=True)
        space =0.1*size[0]
        gs.update(wspace=0.1*size[0],hspace=0.12*size[0],left=0.07,right=0.95,bottom=0.05,top=0.93) #this is still a stupid hardwired parameter
        axes_list=[]
        for (N,sh),coords in zip(num_shape_list,coords_list):
            axes_list.append([fig.add_subplot(gs[co[0]+frame//2:co[0]+frame//2+sh[0],co[1]+frame//2:co[1]+frame//2+sh[1]]) for co in coords]) 
    
        return axes_list,fig,gs

    def draw(self):
        if self.interactive:
            #self.plot_fig.canvas.draw()
            self.pp.draw()
            time.sleep(0.1)
        else:
            self.pp.show()
            
    def plot_error(self):
        if np.array(self.p.plot_error).any():
            # get axis            
            try:
                axis=self.axes_list[self.error_axes_index][self.error_frame]
                # get runtime info
                error = np.array([info['error'].sum(0) for info in self.runtime.iter_info])
                err_fmag = error[:,0]
                err_phot = error[:,1]
                err_exit = error[:,2]
                axis.hold(False)
                fmag = err_fmag/np.max(err_fmag)
                axis.plot(fmag,label='err_fmag %2.2f%% of %.2e' % (fmag[-1]*100,np.max(err_fmag)))
                axis.hold(True)
                phot = err_phot/np.max(err_phot)
                axis.plot(phot,label='err_phot %2.2f%% of %.2e' % (phot[-1]*100,np.max(err_phot)))
                ex = err_exit/np.max(err_exit)
                axis.plot(ex,label='err_exit %2.2f%% of %.2e' % (ex[-1]*100,np.max(err_exit)))
                axis.legend(loc=1,fontsize=10) #('err_fmag %.2e' % np.max(err_fmag),'err_phot %.2e' % np.max(err_phot),'err_exit %.2e' % np.max(err_exit)),
                self.pp.setp(axis.get_xticklabels(), fontsize=10)
                self.pp.setp(axis.get_yticklabels(), fontsize=10)
            except:
                pass
            
    def plot_storage(self,storage,title="",typ='obj'):
        # get plotting paramters
        pp=storage.plot
        axes=self.axes_list[pp.axes_index]
        weight=pp.get('weight')
        # plotting mask for ramp removal
        sh=storage.data.shape[-2:]
        x,y=np.indices(sh)-np.reshape(np.array(sh)//2,(len(sh),)+len(sh)*(1,))
        mask= (x**2+y**2 < 0.1*min(sh)**2)
        pp.mask=mask
        # cropping
        crop=np.array(sh)*np.array(pp.crop)//2
        crop=-crop.astype(int)
        data=u.crop_pad(storage.data,crop,axes=[-2,-1])
        plot_mask=u.crop_pad(mask,crop,axes=[-2,-1])
        for ii,ind in enumerate([(l,a) for l in pp.layers for a in pp.auto_display]):
            #print ii, ind
            if ii >= len(axes):
                break
            # get the layer
            dat=data[ind[0]]
            if ind[1]=='p' or ind[1]=='c':
                if pp.rm_pr:                    
                    if weight is None:
                        ndat = u.rmphaseramp(dat, np.abs(dat) * plot_mask.astype(float))
                        mean_ndat = (ndat*plot_mask).sum() / plot_mask.sum()
                    else:
                        ndat = u.rmphaseramp(dat, np.abs(dat) * weight)
                        mean_ndat = (ndat*weight).sum() / weight.sum()
                else:
                    ndat=dat.copy()
                    mean_ndat = (ndat*plot_mask).sum() / plot_mask.sum()
            else:
                ndat=dat.copy()
            
            if typ=='obj':
                mm = np.mean(np.abs(ndat*plot_mask)**2)
                info = 'T=%.2f' % mm
            else:
                mm = np.sum(np.abs(ndat)**2)
                info = 'P=%1.1e' % mm
                
            if ind[1]=='c':
                dat_i = u.imsave(np.flipud(ndat))
                if not axes[ii].images:
                    axes[ii].imshow(dat_i)
                    self.pp.setp(axes[ii].get_xticklabels(), fontsize=8)
                    self.pp.setp(axes[ii].get_yticklabels(), fontsize=8)
                else:
                    axes[ii].images[0].set_data(dat_i)            
                axes[ii].set_title('%s#%d (C)\n%s' % (title,ind[0],info),size=12)
                continue
                
            if ind[1]=='p':        
                d = np.angle(ndat / mean_ndat)
                ttl = '%s#%d (P)' % (title,ind[0]) #% (ind[0],ind[1])
                cmap = self.pp.get_cmap(pp.cmaps[1])
                clims = pp.clims[1]
            elif ind[1]=='a':    
                d = np.abs(ndat)
                ttl = '%s#%d (M)\n%s' % (title,ind[0],info)
                cmap = self.pp.get_cmap(pp.cmaps[0])
                clims = pp.clims[0]
            
            vmin = d[plot_mask].min() if clims is None else clims[0]
            vmax = d[plot_mask].max() if clims is None else clims[1]
            if not axes[ii].images:
                axes[ii].imshow(d,vmin=vmin, vmax=vmax,cmap=cmap)
                self.pp.setp(axes[ii].get_xticklabels(), fontsize=8)
                self.pp.setp(axes[ii].get_yticklabels(), fontsize=8)
            else:
                axes[ii].images[0].set_data(d)
                axes[ii].images[0].set_clim(vmin=vmin, vmax=vmax)
            axes[ii].set_title(ttl,size=12)
            #ii+=1
    
    def plot_all(self):
        for key,storage in self.pr.items():
            #print key
            self.plot_storage(storage,str(key),'pr')
        for key,storage in self.ob.items():
            #print key
            self.plot_storage(storage,str(key),'obj')
        self.plot_error()
        
    def loop_plot(self, timeout=0.1):
        self.initialize()
        self.request_data()
        while len(self.client.pending): time.sleep(timeout)
        self.mv_data_from_client_buffer()
        self.update_plot_layout()
        self.request_data()
        while True:
            if len(self.client.pending) != 0:
                # at this point the gui should be able to react again
                u.pause(timeout)
            else:
                self.mv_data_from_client_buffer()
                self.plot_all()
                self.draw()
                self.request_data()
                
C=Plot_Client()#(network_DEFAULT)
C.loop_plot()
#C.get_all_data()
#C.update_plot_layout()
#C.plot_all()
"""
while True:
    C.get_all_data()
    C.plot_all()
    C.draw()
"""
