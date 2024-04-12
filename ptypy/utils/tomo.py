# -*- coding: utf-8 -*-
"""
Numerical util functions.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
import scipy.sparse as sparse
import scipy.ndimage as ndimage
import astra
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import ptypy.utils.tomo as tu
from scipy.ndimage import gaussian_filter


# For creating example
def sample_volume(N):
    vol = np.zeros((N,N,N))
    xx,yy,zz = np.meshgrid(np.arange(N)-N/2, np.arange(N)-N/2,np.arange(N)-N/2)
    m = lambda r,dx,dy,dz: np.sqrt((xx+dx)**2 + (yy+dy)**2 + (zz+dz)**2) < r
    vol[m(N//6,0,0,0)] = 1
    vol[m(N//8,N//9,N//9,N//9)] = 2
    vol[m(N//12,N//12,0,0)] = 5
    vol[m(N//12,0,N//8,0)] = 5
    return ndimage.gaussian_filter(vol,1)

def refractive_index_map(Nx):
    beta = np.log(sample_volume(Nx)+20)-np.log(20)
    delta = 0.05 * sample_volume(Nx)
    return delta + 1j * beta



class AstraTomoWrapper:
    """
    Base class for wrappers for the Astra projectors.
    """    

    def __init__(self, obj, vol, angles, obj_is_refractive_index=False, mask_threshold=0):
        self._obj = obj
        self._vol = vol
        self._angles = angles
        self._obj_is_rindex = obj_is_refractive_index
        self._mask_threshold = mask_threshold
        self._create_proj_geometry()
        self._create_vol_geom_and_ids()

    def _create_astra_proj_array(self):
        self._proj_id_real = astra.data3d.create("-sino", self._proj_geom, self._proj_array.real)
        self._proj_id_imag = astra.data3d.create("-sino", self._proj_geom, self._proj_array.imag)

    def _create_proj_geometry(self):
        raise NotImplementedError("Subclass needs to define this.") 

    def _create_vol_geom_and_ids(self):
        self._vol_geom = astra.create_vol_geom(self._vol.shape[0],self._vol.shape[1],self._vol.shape[2])
        self._vol_id_real = astra.data3d.create("-vol", self._vol_geom, self._vol.real)
        self._vol_id_imag = astra.data3d.create("-vol", self._vol_geom, self._vol.imag)
        return self._vol_geom

    def apply_phase_shift_to_obj(self):

        for key, s in self._obj.storages.items():
            cover = s.get_view_coverage().real > self._mask_threshold
            masked_data = np.angle(s.data)*cover
            self._obj.storages[key].data[:] = s.data * np.exp(-1j * np.median(masked_data[masked_data!=0]))

        ## PLOTTING
        # stacked_views = np.array([v.data for v in self._obj.views.values()])
        # view_i = stacked_views[18, :, :]
        # self.plot_complex_array(view_i, title='View after phase shift on obj.storages')

    def plot_complex_array(self, X, title=''):
        norm = colors.Normalize(-5, 5)
        cmap = cm.get_cmap("Spectral")

        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(6,4), dpi=100)
        limit = 1.9
        im0 = axes[0].imshow(X.real, vmax=limit, vmin=-limit)
        axes[0].set_title(f"Real part")

        im1 = axes[1].imshow(X.imag, vmax=limit, vmin=-limit)
        axes[1].set_title(f"Imag part")

        fig.suptitle(title)
        fig.colorbar(im1, ax=axes.ravel().tolist())
        plt.show()    

    def plot_vol(self, title=''):
        pshape = self._vol.shape[0]
        rmap = tu.refractive_index_map(pshape)
        X = rmap.reshape(pshape, pshape, pshape)
        R = np.real(self._vol)
        I = np.imag(self._vol)
        
        pos_limit = max([np.max(X.real), np.max(R)])
        neg_limit = min([np.min(X.real), np.min(R)])

        fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(6,4), dpi=100)
        for i in range(3):
            for j in range(2):
                ax = axes[j,i]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
        axes[0,0].set_title("slice(Z)")
        axes[0,1].set_title("slice(Y)")
        axes[0,2].set_title("slice(X)")
        axes[0,0].set_ylabel("Original")
        axes[0,0].imshow((X.real)[pshape//2], vmin=neg_limit, vmax=pos_limit)
        axes[0,1].imshow((X.real)[:,pshape//2], vmin=neg_limit, vmax=pos_limit)
        axes[0,2].imshow((X.real)[:,:,pshape//2], vmin=neg_limit, vmax=pos_limit)
        axes[1,0].set_ylabel("Recons")
        axes[1,0].imshow((R)[pshape//2], vmin=neg_limit, vmax=pos_limit)
        axes[1,1].imshow((R)[:,pshape//2], vmin=neg_limit, vmax=pos_limit)
        im1 = axes[1,2].imshow((R)[:,:,pshape//2], vmin=neg_limit, vmax=pos_limit)
        fig.suptitle('Real part, '+ title)
        fig.colorbar(im1, ax=axes.ravel().tolist())
        plt.show()

        pos_limit = max([np.max(X.imag), np.max(I)])
        neg_limit = min([np.min(X.imag), np.min(I)])
        fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(6,4), dpi=100)
        for i in range(3):
            for j in range(2):
                ax = axes[j,i]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
        axes[0,0].set_title("slice(Z)")
        axes[0,1].set_title("slice(Y)")
        axes[0,2].set_title("slice(X)")
        axes[0,0].set_ylabel("Original")
        axes[0,0].imshow((X.imag)[pshape//2], vmin=neg_limit, vmax=pos_limit)
        axes[0,1].imshow((X.imag)[:,pshape//2], vmin=neg_limit, vmax=pos_limit)
        axes[0,2].imshow((X.imag)[:,:,pshape//2], vmin=neg_limit, vmax=pos_limit)
        axes[1,0].set_ylabel("Recons")
        axes[1,0].imshow((I)[pshape//2], vmin=neg_limit, vmax=pos_limit)
        axes[1,1].imshow((I)[:,pshape//2], vmin=neg_limit, vmax=pos_limit)
        im1 = axes[1,2].imshow((I)[:,:,pshape//2], vmin=neg_limit, vmax=pos_limit)

        fig.suptitle('Imag part, '+ title)
        fig.colorbar(im1, ax=axes.ravel().tolist())
        plt.show()



class AstraTomoWrapperViewBased(AstraTomoWrapper):
    """
    Wrapper for the Astra projectors, method ViewBased.
    """

    def _create_proj_geometry(self):
        self._fsh = np.array([v.shape for v in self._obj.views.values()]).max(axis=0)
        self._vec = np.zeros((len(self._obj.views),12))
        for i,(k,v) in enumerate(self._obj.views.items()):

            alpha = self._angles[v.storageID]
            y = v.dcoord[0] - v.storage.center[0]
            x = v.dcoord[1] - v.storage.center[1]

            # ray direction
            self._vec[i,0] = np.sin(alpha)
            self._vec[i,1] = -np.cos(alpha)
            self._vec[i,2] = 0

            # center of detector
            self._vec[i,3] = x * np.cos(alpha)
            self._vec[i,4] = x * np.sin(alpha)
            self._vec[i,5] = y 
            
            # vector from detector pixel (0,0) to (0,1)
            self._vec[i,6] = np.cos(alpha)
            self._vec[i,7] = np.sin(alpha)
            self._vec[i,8] = 0
            
            # vector from detector pixel (0,0) to (1,0)
            self._vec[i,9] = 0
            self._vec[i,10] = 0
            self._vec[i,11] = 1

        self._proj_geom = astra.create_proj_geom('parallel3d_vec',  self._fsh[0], self._fsh[1], self._vec)
        return self._proj_geom


    def do_conversion_and_set_proj_array(self, _obj_is_rindex=False):
        
        stacked_views = np.array([v.data for v in self._obj.views.values()])
        self._proj_array = np.moveaxis(stacked_views, 1, 0)
        if not _obj_is_rindex:
            self._proj_array = np.angle(self._proj_array) - 1j * np.log(np.abs(self._proj_array))
        
        ## PLOTTING
        # view_i = self._proj_array[:, 18, :]
        # self.plot_complex_array(view_i, title='View after log conversion')


    def apply_mask_to_proj_array(self):

        if self._mask_threshold != 0:
            for i,v in enumerate(self._obj.views.values()):
                cover = v.storage.get_view_coverage().real > self._mask_threshold 
                real_part = self._proj_array[:,i,:].real  * cover[v.slice]
                imag_part = self._proj_array[:,i,:].imag * cover[v.slice] 

                self._proj_array[:,i,:] = (real_part + 1j * imag_part)

        ## PLOTTING
        # view_i = self._proj_array[:, 18, :]
        # self.plot_complex_array(view_i, title='View after masking (before tomo)')


    def forward(self, type = "FP3D_CUDA", iter=10):

        cfg = astra.astra_dict(type)
        cfg["VolumeDataId"] = self._vol_id_real
        cfg["ProjectionDataId"] = self._proj_id_real
        alg_id_real = astra.algorithm.create(cfg)

        cfg = astra.astra_dict(type)
        cfg["VolumeDataId"] = self._vol_id_imag
        cfg["ProjectionDataId"] = self._proj_id_imag
        alg_id_imag = astra.algorithm.create(cfg)

        astra.algorithm.run(alg_id_real, iter)
        astra.algorithm.run(alg_id_imag, iter)

        _proj_data_real = astra.data3d.get(self._proj_id_real)
        _proj_data_imag = astra.data3d.get(self._proj_id_imag)

        _ob_views_real = np.moveaxis(_proj_data_real, 0, 1)
        _ob_views_imag = np.moveaxis(_proj_data_imag, 0, 1)
        for i, (k,v) in enumerate(self._obj.views.items()):
            real_part = _ob_views_real[i] 
            imag_part = _ob_views_imag[i]

            _obj = real_part + 1j * imag_part

            ## PLOTTING
            # if i==18:
            #     self.plot_complex_array(_obj, title='View after tomo_update, before exp ')

            if not self._obj_is_rindex:
                _obj = np.exp(1j * _obj) 
            v.data[:] = _obj  

            ## PLOTTING
            # if i==18:
            #     view_i = _obj
            #     self.plot_complex_array(_obj, title='View after tomo_update, after rxp')


    def backward(self, type="BP3D_CUDA", iter=10):
        print('Computing backward projection')
        cfg = astra.astra_dict(type)
        cfg["ReconstructionDataId"] = self._vol_id_real
        cfg["ProjectionDataId"] = self._proj_id_real
        alg_id_real = astra.algorithm.create(cfg)

        cfg = astra.astra_dict(type)
        cfg["ReconstructionDataId"] = self._vol_id_imag
        cfg["ProjectionDataId"] = self._proj_id_imag
        alg_id_imag = astra.algorithm.create(cfg)

        astra.algorithm.run(alg_id_real, iter)
        astra.algorithm.run(alg_id_imag, iter)

        vol_real = astra.data3d.get(self._vol_id_real)
        vol_imag = astra.data3d.get(self._vol_id_imag)
        self._vol = vol_real + 1j * vol_imag

        self.plot_vol(title='Vol after backward')

        return self._vol



class AstraTomoWrapperProjectionBased(AstraTomoWrapper):
    """
    Wrapper for the Astra projectors, method ProjectionBased.
    """

    def _create_proj_geometry(self):
        self._ssh = np.array([s.shape for s in self._obj.storages.values()]).max(axis=0)
        self._vec = np.zeros((len(self._obj.storages),12))
        for i, (storageID, storage) in enumerate(self._obj.storages.items()): 

            alpha = self._angles[storageID]
            y = 0  
            x = 0 

            # ray direction
            self._vec[i,0] = np.sin(alpha)
            self._vec[i,1] = -np.cos(alpha)
            self._vec[i,2] = 0

            # center of detector
            self._vec[i,3] = x * np.cos(alpha)
            self._vec[i,4] = x * np.sin(alpha)
            self._vec[i,5] = y 
            
            # vector from detector pixel (0,0) to (0,1)
            self._vec[i,6] = np.cos(alpha)
            self._vec[i,7] = np.sin(alpha)
            self._vec[i,8] = 0
            
            # vector from detector pixel (0,0) to (1,0)
            self._vec[i,9] = 0
            self._vec[i,10] = 0
            self._vec[i,11] = 1

        self._proj_geom = astra.create_proj_geom('parallel3d_vec',  self._ssh[1], self._ssh[2], self._vec)
        return self._proj_geom

    def do_conversion_and_set_proj_array(self, _obj_is_rindex=False):
        print('Taking log of proj array')
        self._proj_array = np.array([s.data for s in self._obj.storages.values()])
        
        if not _obj_is_rindex:
            self._proj_array = np.angle(self._proj_array) - 1j * np.log(np.abs(self._proj_array))
        
        self._proj_array = np.squeeze(self._proj_array)
        self._proj_array = np.moveaxis(self._proj_array, 1, 0)

    def apply_mask_to_proj_array(self):

        if self._mask_threshold != 0:
            for i, s in enumerate(self._obj.storages.values()):
                cover = s.get_view_coverage().real > self._mask_threshold 
                real_part = self._proj_array[:,i,:].real  * cover
                imag_part = self._proj_array[:,i,:].imag * cover

                self._proj_array[:,i,:] = (real_part + 1j * imag_part)

        ## PLOTTING
        # scan_i = self._proj_array[:, 18, :]
        # self.plot_complex_array(scan_i, title='Scan after masking (before tomo)')

    def forward(self, type = "FP3D_CUDA", iter=10):
        print('Computing forward projection')
        cfg = astra.astra_dict(type)
        cfg["VolumeDataId"] = self._vol_id_real
        cfg["ProjectionDataId"] = self._proj_id_real
        alg_id_real = astra.algorithm.create(cfg)

        cfg = astra.astra_dict(type)
        cfg["VolumeDataId"] = self._vol_id_imag
        cfg["ProjectionDataId"] = self._proj_id_imag
        alg_id_imag = astra.algorithm.create(cfg)

        astra.algorithm.run(alg_id_real, iter)
        astra.algorithm.run(alg_id_imag, iter)

        _proj_data_real = astra.data3d.get(self._proj_id_real)
        _proj_data_imag = astra.data3d.get(self._proj_id_imag)

        _ob_views_real = np.moveaxis(_proj_data_real, 0, 1)
        _ob_views_imag = np.moveaxis(_proj_data_imag, 0, 1)
        for i, (k,v) in enumerate(self._obj.storages.items()):
            _obj = _ob_views_real[i] + 1j*_ob_views_imag[i]

            ## PLOTTING
            # if i==18:
            #     self.plot_complex_array(_obj, title='Scan after tomo_update, before exp ')

            if not self._obj_is_rindex:
                _obj = np.exp(1j * _obj) 
            v.data[:] = _obj 

            ## PLOTTING
            # if i==18:
            #     self.plot_complex_array(_obj, title='Scan after tomo_update, after exp ')
        
    def backward(self, type="BP3D_CUDA", iter=10):
        print('Computing backward projection')
        cfg = astra.astra_dict(type)
        cfg["ReconstructionDataId"] = self._vol_id_real
        cfg["ProjectionDataId"] = self._proj_id_real
        alg_id_real = astra.algorithm.create(cfg)

        cfg = astra.astra_dict(type)
        cfg["ReconstructionDataId"] = self._vol_id_imag
        cfg["ProjectionDataId"] = self._proj_id_imag
        alg_id_imag = astra.algorithm.create(cfg)

        astra.algorithm.run(alg_id_real, iter)
        astra.algorithm.run(alg_id_imag, iter)

        vol_real = astra.data3d.get(self._vol_id_real)
        vol_imag = astra.data3d.get(self._vol_id_imag)
        self._vol = vol_real + 1j * vol_imag

        self.plot_vol()

        return self._vol

