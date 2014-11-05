# -*- coding: utf-8 -*-
"""\
Plotting tool.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import numpy as np
from numpy import fft as FFT
from matplotlib import gridspec
from matplotlib import pyplot as plt

from .. import utils as u
from interaction import Client

__all__ =['Plotter']

DEFAULT = dict(
            figsize = (12,10),
            object_cmaps = ['bone','jet'],
            object_clims = [None,None],
            probe_cmaps = ['hot','hsv'],
            plot_local_error = None,        # [None, 'a','p'] on what to plot a local error map 
            crop_object = [0.25,0.25],      # [None, [fraction_of_probe_rows,fraction_of_probe_cols]] 
            crop_probe = [0.0,0.0],       # [None, [fraction_of_diff_pattern_rows,fraction_of_diff_pattern_cols]] 
            remove_phase_ramp = True,
            object_frame_shape = None,      # [None, (rows,cols)] takes tuple, if None takes info from p
            object_frame_nums = None,       # [None, num]            
            object_index_list = None,       # selected objects, otherwise its auto 
            object_auto_display = ['a','p'], # list of data to derive from each object, choose from 'a','p' and 'c'
            probe_frame_shape = None,        # [None, (rows,cols)] takes tuple
            probe_frame_nums = None,        # [None, num]
            probe_index_list = None,        # selected probes
            probe_auto_display = ['c'],      # list of data to derive from each probe, choose from 'a','p' and 'c'
            simplified_aspect_ratios=True,
            gridspecpars=(0.1,0.12,0.07,0.95,0.05,0.93),
)

templates= dict(
    minimal = dict(
        figsize = (8,8),
        object_cmaps = ['bone','jet'],
        probe_cmaps = ['hot','hsv'],
        plot_local_error = None,    # # [None, 'a','p'] on what to plot a local error map 
        crop_object = [0.25,0.25],    # [None, [fraction_of_probe_rows,fraction_of_probe_cols]]
        remove_phase_ramp = True, 
        object_frame_shape = None,   # [None, (rows,cols)] takes tuple, if None takes info from p
        object_frame_nums = 2,    # [None, num]
        object_index_list = [[0,0,'a'],[0,0,'p']],    # selected objects, otherwise its auto 
        probe_frame_shape = None,    # [None, (rows,cols)] takes tuple
        probe_frame_nums = 2,     # [None, num]
        probe_index_list = [[0,0,'a'],[0,0,'p']],     # selected probes
        simplified_aspect_ratios=True 
    ),
    legacy = dict(
        figsize = (8,6),
        object_cmaps = ['bone','jet'],
        probe_cmaps = ['hot','hsv'],
        plot_local_error = None,    # [None, 'a','p'] on what to plot a local error map 
        crop_object = None,    # [None, [fraction_of_probe_rows,fraction_of_probe_cols]]
        remove_phase_ramp = True, 
        object_frame_shape = (2,2),   # [None, (rows,cols)] takes tuple, if None takes info from p
        object_frame_nums = 2,    # [None, num]
        object_index_list = [[0,0,'a'],[0,0,'p']],    # selected objects, otherwise its auto 
        probe_frame_shape = (2,2),    # [None, (rows,cols)] takes tuple
        probe_frame_nums = 2,     # [None, num]
        probe_index_list = [[0,0,'c']],     # selected probes
        simplified_aspect_ratios=False,
        gridspecpars=(0.2,0.2,0.1,0.9,0.1,0.9)  
    ),
    default = default_template
)
"""\
class Plot_Client(Client):
    
    DEFAULT_PLOT = 
    
    def __init__(self, client_pars=None, plot_template=None, interactive=True, **kwargs)
        
"""
   
class Plotter(object):

    def __init__(self, pdict=None, plot_template=None, interactive=True, **kwargs):
        from matplotlib import pyplot
        self.interactive = interactive
        self.pp = pyplot
        pyplot.interactive(interactive)
        
        #self.template_default = default_template
        self.templates = templates
        self.params = U.Param()
        self.params.update(self.templates['default'])
        self.update_plot_layout(pdict=pdict,plot_template=plot_template)

    def update_plot_layout(self,pdict=None,plot_template=None,**kwargs):
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
        
        p=U.Param(pdict)
        # try to get info from p
        ptemplate = p.get('plot_template')
        # print ptemplate
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
                
            self.params.update(template)

        if self.params.object_frame_nums is None:
            no=p.obj.shape[:-2]
            if len(no)==0:
                obj_nums=1
            elif len(no)==1:
                obj_nums=no[0]
            elif len(no)==2:
                obj_nums=no[0]*no[1]
            obj_nums*=len(self.params.object_auto_display)
        else:
            obj_nums=self.params.object_frame_nums

        if self.params.probe_frame_nums is None:    
            np=p.probe.shape[:-2]
            if len(np)==0:
                pr_nums=1
            elif len(np)==1:
                pr_nums=np[0]
            elif len(np)==2:
                pr_nums=np[0]*np[1]
            pr_nums*=len(self.params.probe_auto_display)
        else:
            pr_nums=self.params.probe_frame_nums
                        
        if self.params.object_frame_shape is None:
            obj_sh=p.obj.shape[-2:]
            if self.params.simplified_aspect_ratios:
                obj_sh = simplify_aspect_ratios(obj_sh)
        else:
            obj_sh=self.params.object_frame_shape
        


        if self.params.probe_frame_shape is None:
            pr_sh=p.probe.shape[-2:]
            if self.params.simplified_aspect_ratios:
                pr_sh = (2,2)
        else:
            pr_sh=self.params.probe_frame_shape
            
        #print obj_sh
        #if self.params.crop_object is not None and self.params.probe_frame_shape is None and self.params.object_frame_shape is None:
        #    self.crop_object = [int(self.params.crop_object[0]*pr_sh[0]),int(self.params.crop_object[1]*pr_sh[1])]
        #    obj_sh=(obj_sh[0]-self.crop_object[0],obj_sh[1]-self.crop_object[1])
            
        
        #num_list=[obj_nums,pr_nums]
        if self.params.plot_local_error is not None:
            lerr_axes_num = p.get('Nscan')
            num_list=[obj_nums+lerr_axes_num,pr_nums+1] # the +1 is for the error
        else:
            lerr_axes_num = 0
            num_list=[obj_nums,pr_nums+1]
            
        shape_list=[obj_sh,pr_sh]
        #print shape_list
        plot_axes_list,plot_fig,gs = self.create_plot_from_tile_list(1,shape_list,num_list,self.params.figsize)
        #plot_fig.suptitle(p.paramdict.get('scans')[0])
        obj_axes = plot_axes_list[0]
        pr_axes = plot_axes_list[1]
        err_axes = [pr_axes.pop(0)]
        local_error_axes=[obj_axes.pop(0) for i in range(lerr_axes_num)]
        sy,sx = gs.get_geometry()
        w,h,l,r,b,t = self.params.gridspecpars
        gs.update(wspace=w*sx,hspace=h*sx,left=l,right=r,bottom=b,top=t)
        plot_axes = obj_axes+pr_axes+err_axes
        plot_fig.hold(False)
        for pl in obj_axes+pr_axes+err_axes: 
            pl.hold(False)
            self.pp.setp(pl.get_xticklabels(), fontsize=8) #doesn't do nothin
            self.pp.setp(pl.get_yticklabels(), fontsize=8)
        self.plot_fig = plot_fig
        self.obj_axes = obj_axes 
        self.pr_axes = pr_axes
        self.err_axes = err_axes
        self.local_error_axes = local_error_axes        
        self.gs= gs
        
    def plot(self, pdict=None, pr_ind=None, obj_ind=None, **kwargs):
        """\
        Plots ptycho results, drawing necessary variables from 
        dictionnary p and/or keywords arguments. Passing 
        globals() works.
        pri,obi are the indices of probe and object to show if there are more than one.
        (default: 0)
        """
        p = U.Param(pdict)
        if pr_ind is None:
            pr_ind =self.params.probe_index_list
        
        if obj_ind is None:
            obj_ind =self.params.object_index_list
            
        err = p.get('LL_list')
        if err is None:
            err = p.get('err')
            if err is None:
                err = [0]
            else:
                err_label = 'Error'
            obj = p.get('object')
            if obj is None:
                obj = p.obj
        else:
            err_label = 'Log-likelihood'
            obj = p.obj

        probe = p.probe
        
        plot_mask = p.get('plot_mask')
        if plot_mask is None:
            if p.nearfield:
                plot_mask = np.ones(obj_sh)
            else:
                plot_mask = np.fft.fftshift(U.fvec2(obj_sh)) < .25*(min(obj_sh) - max(probe.shape))**2
        #print obj_sh, probe.shape 
        #print plot_mask[plot_mask]
        obj_weight = p.get('pr_nrm')
        #roi cropping
        pr_sh = probe.shape[-2:]
        if self.params.crop_object is not None:
            crop = [int(self.params.crop_object[0]*pr_sh[0]),int(self.params.crop_object[1]*pr_sh[1])]
            obj=obj[...,crop[0]:-1-crop[0],crop[1]:-1-crop[1]]
            if obj_weight is not None:
                obj_weight=obj_weight[...,crop[0]:-1-crop[0],crop[1]:-1-crop[1]]
            plot_mask=plot_mask[...,crop[0]:-1-crop[0],crop[1]:-1-crop[1]]

        if self.params.crop_probe is not None:
            crop2 = [int(self.params.crop_probe[0]*pr_sh[0]),int(self.params.crop_probe[1]*pr_sh[1])]
            probe=probe[...,crop2[0]:-1-crop2[0],crop2[1]:-1-crop2[1]]
            
        # autodetection of probes abnd objects to diplay and generate the lists.
        osh = obj.shape
        psh = probe.shape
        if obj_ind is None:
            if obj.ndim == 3:
                obj_ind = [[i,0,s] for s in self.params.object_auto_display for i in range(osh[0]) ]
            elif obj.ndim == 4:
                obj_ind = [[i,j,s] for s in self.params.object_auto_display for i in range(osh[0]) for j in range(osh[1]) ]
        if pr_ind is None:
            if probe.ndim == 3:
                pr_ind = [[i,0,s] for s in self.params.probe_auto_display for i in range(psh[0])]
            elif probe.ndim == 4:
                pr_ind = [[i,j,s] for i in range(psh[0]) for j in range(psh[1]) for s in self.params.probe_auto_display]
        
        #print pr_ind
        #print obj_ind

        for ii,ind in enumerate(obj_ind):
            if ii >= len(self.obj_axes):
                break
            #print ind
            if len(osh[:-2])==0:
                ob=obj
                if obj_weight is not None:
                    objw=obj_weight
            elif len(osh[:-2])==1:
                ob=obj[ind[0]]
                if obj_weight is not None:
                    objw=obj_weight[ind[0]]
            elif len(osh[:-2])==2:
                ob=obj[ind[0],ind[1]]
                if obj_weight is not None:
                    objw=obj_weight[ind[0],ind[1]]

            if ind[2]=='p' or ind[2]=='c':
                if self.params.remove_phase_ramp:                    
                    if obj_weight is None:
                        nobj = U.rmphaseramp(ob, np.abs(ob) * plot_mask.astype(float))
                        mean_nobj = (nobj*plot_mask).sum() / plot_mask.sum()
                    else:
                        nobj = U.rmphaseramp(ob, np.abs(ob) * obj_w)
                        mean_nobj = (nobj*objw).sum() / objw.sum()
                else:
                    nobj=ob.copy()
                    mean_nobj = (nobj*plot_mask).sum() / plot_mask.sum()
            else:
                nobj=ob.copy()
            
            transparency = np.mean(np.abs(nobj)**2)
            if ind[2]=='c':
                object_i = U.imsave(np.flipud(nobj))
                if not self.obj_axes[ii].images:
                    self.obj_axes[ii].imshow(object_i)
                    self.pp.setp(self.obj_axes[ii].get_xticklabels(), fontsize=8)
                    self.pp.setp(self.obj_axes[ii].get_yticklabels(), fontsize=8)
                else:
                    self.obj_axes[ii].images[0].set_data(object_i)            
                self.obj_axes[ii].set_title('Object %d-%d Complex, T=%.2f' % (ind[0],ind[1],transparency),size=12)
                continue
                
            if ind[2]=='p':        
                data = np.angle(nobj / mean_nobj)
                ttl = 'Object %d-%d Phase' % (ind[0],ind[1])
                cmap = self.pp.get_cmap(self.params.object_cmaps[1])
                clims = self.params.object_clims[1]
            elif ind[2]=='a':    
                data = np.abs(nobj)
                ttl = 'Object %d-%d Modulus, T=%.2f' % (ind[0],ind[1],transparency)
                cmap = self.pp.get_cmap(self.params.object_cmaps[0])
                clims = self.params.object_clims[0]
            
            vmin = data[plot_mask].min() if clims is None else clims[0]
            vmax = data[plot_mask].max() if clims is None else clims[1]
            if not self.obj_axes[ii].images:
                self.obj_axes[ii].imshow(data,vmin=vmin, vmax=vmax,cmap=cmap)
                self.pp.setp(self.obj_axes[ii].get_xticklabels(), fontsize=8)
                self.pp.setp(self.obj_axes[ii].get_yticklabels(), fontsize=8)
            else:
                self.obj_axes[ii].images[0].set_data(data)
                self.obj_axes[ii].images[0].set_clim(vmin=vmin, vmax=vmax)
            self.obj_axes[ii].set_title(ttl,size=12)
            ii+=1
            
        for ii,ind in enumerate(pr_ind):
            if ii >= len(self.pr_axes):
                break
            if len(psh[:-2])==0:
                pr=probe
                
            elif len(psh[:-2])==1:
                pr=probe[ind[0]]
            elif len(psh[:-2])==2:
                pr=probe[ind[0],ind[1]] 
                
            power=np.sum(np.abs(pr)**2)
            # flip probe to be consistent with usual imshow convention
            pr=np.flipud(pr)
                
            if ind[2]=='c':
                probe_i = U.imsave(pr)
                ttl = 'Probe %d-%d c, P=%1.1e' % (ind[0],ind[1],power)
            elif ind[2]=='a':
                cmap = self.pp.get_cmap(self.params.probe_cmaps[0])
                probe_i = U.imsave(np.abs(pr),cmap=cmap)
                ttl = 'Probe %d-%d m' % (ind[0],ind[1])
            elif ind[2]=='p':
                cmap = self.pp.get_cmap(self.params.probe_cmaps[1])
                ttl = 'Probe %d-%d p' % (ind[0],ind[1])
                probe_i = U.imsave(np.angle(pr),cmap=cmap)
                
            if not self.pr_axes[ii].images:
                self.pr_axes[ii].imshow(probe_i)
                self.pp.setp(self.pr_axes[ii].get_xticklabels(), fontsize=8)
                self.pp.setp(self.pr_axes[ii].get_yticklabels(), fontsize=8)
            else:
                self.pr_axes[ii].images[0].set_data(probe_i)            
            self.pr_axes[ii].set_title(ttl,size=10)
            
            ii+=1
        """
        lerr=p.get('err_local')
        if self.params.plot_local_error is not None and lerr is not None:
            pos=p.get('positions')
            Ndata_scan=p.get('Ndata_scan')
            Nscan=p.get('Nscan')
            
            pdiffs = kwargs.get('pos_diffs')
            if pdiffs is None:
                pdiffs=np.zeros((Ndata_scan,2))
                                     
            if self.params.plot_local_error=='p':        
                data = np.angle(nobj / mean_nobj)
                cmap = self.pp.get_cmap(self.params.object_cmaps[1])
                clims = self.params.object_clims[1]
            else:    
                data = np.abs(nobj)
                cmap = self.pp.get_cmap(self.params.object_cmaps[0])
                clims = self.params.object_clims[0]
            
            vmin = data[plot_mask].min() if clims is None else clims[0]
            vmax = data[plot_mask].max() if clims is None else clims[1]
           
            for ii in range(len(self.local_error_axes)):
                errlist=lerr[ii*Ndata_scan:(ii+1)*Ndata_scan]
                posis=pos[ii*Ndata_scan:(ii+1)*Ndata_scan]
                xscat=posis[:,1]+posis[:,5]+pdiffs[:,1]
                yscat=posis[:,0]+posis[:,4]+pdiffs[:,0]
                
                mn=np.min(errlist)
                mx=np.max(errlist)
                mnn=np.mean(errlist)
                self.local_error_axes[ii].cla()
                # plotting background image
                if not self.local_error_axes[ii].images:
                    self.local_error_axes[ii].imshow(data,cmap=cmap)
                    self.pp.setp(self.local_error_axes[ii].get_xticklabels(), fontsize=8)
                    self.pp.setp(self.local_error_axes[ii].get_yticklabels(), fontsize=8)
                else:
                    self.local_error_axes[ii].images[0].set_data(data)
                    self.local_error_axes[ii].images[0].set_clim(vmin=vmin, vmax=vmax)
                # estimate scan point density
                disp_pos=self.local_error_axes[ii].transData.transform(posis[:,:2])
                point_size=2*np.sqrt(np.prod(disp_pos.max(0)-disp_pos.min(0))/len(posis))
                
                self.local_error_axes[ii].set_title('(%d,%d) mean=%.4f, max=%.4f, min=%.4f' % (posis[0,2],posis[0,3],mnn,mx,mn), size=10)
                sc=self.local_error_axes[ii].scatter(xscat+pr_sh[1]/2-crop[1],yscat+pr_sh[0]/2-crop[0],s=point_size,c=errlist,cmap='jet',linewidths=0)
                self.local_error_axes[ii].axis([0,data.shape[1],0,data.shape[0]])
                self.local_error_axes[ii].invert_yaxis()
            """
        self.err_axes[0].plot(err)
        self.err_axes[0].set_title(err_label)
        self.pp.setp(self.err_axes[0].get_xticklabels(), fontsize=10)
        self.pp.setp(self.err_axes[0].get_yticklabels(), fontsize=10)
        self.plot_fig.canvas.set_window_title(p.run_name)
        return

    def draw(self):
        if self.interactive:
            #self.plot_fig.canvas.draw()
            self.pp.draw()
            U.pause(0.01)
        else:
            self.pp.show()

    def savefig(self, *args, **kwargs):
        self.plot_fig.savefig(*args, **kwargs)
        
    def create_plot_from_tile_list(self,fignum=1,shape_list=[(2,2)],num_list=[4],figsize=(8,8)):        
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
                    coords=[(size[0],ii*sh[1]+a) for ii in range(N)]
                    size[0]+=sh[0]
                else:
                    N=N_v
                    a=size[0]%sh[0]
                    coords=[(ii*sh[0]+a,size[1]) for ii in range(N)]
                    size[1]+=sh[1]
                    
                num -=N
                coords_tl+=coords
                coords_tl.sort()
                
            return coords_tl, size
        
        coords_list=[]
        fig_aspect_ratio = figsize[0]/float(figsize[1])
        size=[0,0] 
        for N,sh in zip(num_list,shape_list):
            coords, size  = fill_with_tiles(size,sh,N,fig_aspect_ratio)
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
        for sh,coords in zip(shape_list,coords_list):
            axes_list.append([fig.add_subplot(gs[co[0]:co[0]+sh[0],co[1]:co[1]+sh[1]]) for co in coords]) 
    
        return axes_list,fig,gs

