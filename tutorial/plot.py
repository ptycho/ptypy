from ptypy.resources import flower_obj as flower
from ptypy.resources import moon_pr as moon
import numpy as np
from ptypy import utils as u
import matplotlib as mpl
plt = mpl.pyplot
plt.close('all')



class ManagedAxis(object):
    def __init__(self,ax=None, data = None, channel='a',cmap = None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        self.ax = ax
        self.shape =None
        self.set_channel(channel,False)
        self.set_cmap(cmap, False)
        self.remove_phase_ramp = True
        self.cax = None
        self.vmin = None
        self.vmax = None
        self.mn = None
        self.mx = None
        self.mask = None
        
    def set_psize(self,psize, update=True):
        assert np.isscalar(psize) ,'Pixel size must be scalar value'
        self.psize=np.abs(psize)
        if update:
            self._update()
    
    def set_channel(self,channel, update=True):
        assert channel in ['a','c','p'], 'Channel must be either (a)bs, (p)hase or (c)omplex'
        self.channel=channel
        if update:
            self._update()
            self._update_colorscale()
            
    def set_cmap(self,cmap, update=True):
        try:
            self.cmap = mpl.cm.get_cmap(cmap)
        except:
            print "Colormap `%s` not found. Using `gray`" %str(cmap)
            self.cmap = mpl.cm.get_cmap('gray')
        if update:
            self._update()
            self._update_colorscale()
            
    def set_clim(self,vmin,vmax, update=True):
        self.vmin=vmin
        self.vmax=vmax
        assert vmin<vmax
        if update:
            self._update()
            
    def set_mask(self,mask, update=True):
        if mask is not None:
            if np.isscalar(mask) and self.shape is not None:
                x,y=u.grids(self.shape)
                self.mask = (np.sqrt(x**2+y**2)<np.abs(mask))
            else:
                self.mask = mask
                if self.shape is None:
                    self.shape = self.mask.shape
                else:
                    assert self.shape == self.mask.shape
        else:
            self.mask=None
        if update:
            self._update()
    
    def set_data(self,data):
        assert data.ndim ==2 ,'Data must be two dimensional. It is %d-dimensional' % data.ndim
        self.data = data 
        self.shape = self.data.shape
        self._update()
        
    def _update(self):
        if str(self.channel)=='a':
            imdata = np.abs(self.data)
        elif str(self.channel)=='p':
            if self.remove_phase_ramp:
                imdata = np.angle(u.rmphaseramp(self.data)) 
            else:
                imdata = np.angle(self.data)
        elif str(self.channel)=='c':
            if self.remove_phase_ramp:
                imdata = u.rmphaseramp(self.data)
        else:
            imdata = np.abs(self.data)
        
        if self.mask is not None:
            self.mx = np.max(self.mask* np.abs(imdata))
            if self.vmax is None or self.mx<self.vmax:
                mx = self.mx 
            self.mn = np.min(self.mask* np.abs(imdata))
            if self.vmin is None or self.mn>self.vmin:
                mn = self.mn
        else:
            mn,mx = self.vmin,self.vmax
        
        pilim = u.imsave(imdata,cmap=self.cmap,vmin=mn,vmax=mn) 
        if not self.ax.images:
            self.ax.imshow(pilim)
            plt.setp(self.ax.get_xticklabels(), fontsize=8)
            plt.setp(self.ax.get_yticklabels(), fontsize=8)
        else:
            self.ax.images[0].set_data(pilim)
             
        self._update_colorbar(mn,mx)
        plt.draw()

    def _update_colorscale(self,resolution=256):
        if self.cax is None:
            return
        sh = (resolution,int(resolution/self.cax_aspect))
        psize = (1./sh[0],1./sh[1])
        cax = self.cax
        cax.cla()
        ver, hor = np.indices(sh) * np.asarray(psize).reshape( (len(sh),) + len(sh)*(1,))
        if str(self.channel)=='c':
            comcax = hor *np.exp(2j*np.pi*ver)
            cax.imshow(u.imsave(comcax),extent=[0,1,0,1], aspect=self.cax_aspect )
        else:
            cax.imshow(ver,cmap=self.cmap,extent=[0,1,0,1],aspect=self.cax_aspect )
        #self.cax.axis('tight')
        self.cax.invert_yaxis()
        self._update_colorbar()
        plt.draw()
        
    def _update_colorbar(self,mn=0,mx=1):
        if self.cax is None:
            return
        a = self.ax.get_position().bounds
        b = self.cax.get_position().bounds
        self.cax.set_position((b[0],a[1],b[2],a[3]))
        
    def add_colorbar(self, aspect =10,fraction= 0.15, pad = 0.02,resolution=256):
        if str(self.channel)=='c':
            aspect/=1.5
        self.cax_aspect = aspect
        cax, kw = mpl.colorbar.make_axes_gridspec(self.ax, aspect = aspect,fraction= fraction, pad =pad)
        cax.yaxis.tick_right()
        self.cax = cax
        self._update_colorscale()
    
        plt.draw()

class coloraxis(object):
    
    def __init__(self, axis, aspect =10,fraction= 0.15, pad = 0.02,\
                cmap='complex',vmin=0,vmax=1,resolution=256):
        self.pax = axis
        if str(cmap)=='complex':
            aspect/=1.5
        cax, kw = mpl.colorbar.make_axes_gridspec(axis, aspect = aspect,fraction= fraction, pad =pad)
        cax.yaxis.tick_right()
        self.cax = cax
        sh = (resolution,int(resolution/aspect))
        psize = (1./sh[0],1./sh[1])
        ver, hor = np.indices(sh) * np.asarray(psize).reshape( (len(sh),) + len(sh)*(1,))
        self.vertical = ver
        self.horizontal = hor
        self.vmin = vmin if vmin is not None else 0.
        self.vmax = vmax if vmin is not None else 1.
        if cmap=='complex':
            self.cmap = cmap
            comcax = self.horizontal *np.exp(2j*np.pi*self.vertical)
            self.cax.imshow(u.imsave(comcax))
        else:
            self.cmap = mpl.cm.get_cmap(cmap)
            self.cax.imshow(self.vertical,cmap=self.cmap)
        
        #self.cax.axis('tight')
        self.cax.invert_yaxis
        self._update(vmin,vmax)
        
    def _update(self,vmin=None,vmax=None,cmap=None):
        if vmin is not None:
            self.vmin=vmin
        if vmax is not None:
            self.vmax=vmax

        a = self.pax.get_position().bounds
        b = self.cax.get_position().bounds
        self.cax.set_position((b[0],a[1],b[2],a[3]))

MA = ManagedAxis()
MA.set_data(flower(800)) 
plt.show()
