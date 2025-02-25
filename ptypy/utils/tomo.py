# -*- coding: utf-8 -*-
"""
Numerical util functions.

This file is part of the PTYPY package.

    :copyright: Copyright 2024 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import os
import csv

import numpy as np
import scipy.sparse as sparse
import scipy.ndimage as ndimage
import astra
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import ptypy.utils.tomo as tu


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

    def __init__(self, obj, vol, shifts=None, obj_is_refractive_index=False, mask_threshold=0):
        self._obj = obj
        self._vol = vol
        self._shifts_per_angle = shifts
        self._obj_is_rindex = obj_is_refractive_index
        self._mask_threshold = mask_threshold
        self._create_vol_geom_and_ids()

    def _create_proj_array_and_ids(self):
        n_views = len([v for v in self._obj.views.values() if v.pod.active])
        view_shape = list(self._obj.views.values())[0].shape
        proj_array_shape = (n_views, view_shape[0], view_shape[1])
        empty_proj_array = np.zeros(proj_array_shape, dtype=np.complex64)
        self._proj_array = np.moveaxis(empty_proj_array, 1, 0)
        self._proj_id_real = astra.data3d.create("-sino", self._proj_geom, self._proj_array.real)
        self._proj_id_imag = astra.data3d.create("-sino", self._proj_geom, self._proj_array.imag)

    def _create_proj_geometry(self):
        raise NotImplementedError("Subclass needs to define this.") 

    def _create_vol_geom_and_ids(self):
        self._vol_geom = astra.create_vol_geom(self._vol.shape[0], self._vol.shape[1], self._vol.shape[2])
        self._vol_id_real = astra.data3d.create("-vol", self._vol_geom, self._vol.real)
        self._vol_id_imag = astra.data3d.create("-vol", self._vol_geom, self._vol.imag)

    def plot_complex_array(X, title=''):
        norm = colors.Normalize(-5, 5)
        cmap = cm.get_cmap("Spectral")

        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(6,4), dpi=100)
        max_lim = max(np.max(X.real), np.max(X.imag))
        min_lim = min(np.min(X.real), np.min(X.imag))

        im0 = axes[0].imshow(X.real, vmax=max_lim, vmin=min_lim)
        axes[0].set_title(f"Real part")

        im1 = axes[1].imshow(X.imag, vmax=max_lim, vmin=min_lim)
        axes[1].set_title(f"Imag part")

        fig.suptitle(title)
        fig.colorbar(im1, ax=axes.ravel().tolist())
        plt.show() 

    def plot_vol(self, vol, title=''):
        pshape = vol.shape[0]
        rmap = tu.refractive_index_map(pshape)
        X = rmap.reshape(pshape, pshape, pshape)
        R = np.real(vol)
        I = np.imag(vol)

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
        fig.suptitle('Real part')
        fig.colorbar(im1, ax=axes.ravel().tolist())
        plt.show()
        plt.savefig('real_vol_'+title+'.png')

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

        fig.suptitle('Imag part')
        fig.colorbar(im1, ax=axes.ravel().tolist())
        plt.show()
        plt.savefig('imag_vol_'+title+'.png')
   
   
    def plot_vol_only_recons(self, vol, iter, title=''):
        pshape = vol.shape[0]
        rmap = tu.refractive_index_map(pshape)

        R = np.real(vol)
        I = np.imag(vol)
        pos_limit = max([np.max(R), np.max(I)])  
        neg_limit = min([np.min(R), np.min(I)])  
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
        axes[0,0].set_ylabel("Real")
        axes[0,0].imshow((R)[pshape//2], vmin=neg_limit, vmax=pos_limit)
        axes[0,1].imshow((R)[:,pshape//2], vmin=neg_limit, vmax=pos_limit)
        axes[0,2].imshow((R)[:,:,pshape//2], vmin=neg_limit, vmax=pos_limit)
        axes[1,0].set_ylabel("Imag")
        axes[1,0].imshow((I)[pshape//2], vmin=neg_limit, vmax=pos_limit)
        axes[1,1].imshow((I)[:,pshape//2], vmin=neg_limit, vmax=pos_limit)
        im1 = axes[1,2].imshow((I)[:,:,pshape//2], vmin=neg_limit, vmax=pos_limit)

        fig.suptitle('Recons, iter {}, '.format(iter)+ title)
        fig.colorbar(im1, ax=axes.ravel().tolist())
        plt.show()

class AstraTomoWrapperViewBased(AstraTomoWrapper):
    """
    Wrapper for the Astra projectors, method ViewBased.
    """

    def _create_proj_geometry(self):
        self._fsh = np.array([v.shape for v in self._obj.views.values() if v.pod.active]).max(axis=0)
        self._vec = np.zeros((len([v for v in self._obj.views.values() if v.pod.active]),12))

        # vals_to_save = []
        # i=0
        for i,(k,v) in enumerate([(i,v) for i,v in self._obj.views.items() if v.pod.active]):

            alpha = v.extra 
            # # Save the geometry values for a certain angle
            # if i==0:
            #     ref_id = v.storageID
            # if v.storageID == ref_id:
            #     vals_to_save.append({
            #         's_id': v.storageID,
            #         'v.storage.center[0]':v.storage.center[0],
            #         'v.storage.center[1]':v.storage.center[1],
            #     })
            
            # Apply shifts if they are provided
            if self._shifts_per_angle is not None:
                shift_dx, shift_dy = self._shifts_per_angle[alpha]
                corrected_shift_dx, corrected_shift_dy = shift_dx+10, shift_dy+10
            
                # Hardcoding v.storage.center with 220,220 - needed for real data
                # FIXME: storage centre shouldn't be harcoded here
                # The shifts are also only needed for working on real data
                y = v.dcoord[0] - 220 + corrected_shift_dy   # v.storage.center[0]
                x = v.dcoord[1] - 220 + corrected_shift_dx   # v.storage.center[1]

            else: # no shift is applied
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
        #     i+=1

        # # Write geometry values to file
        # if not os.path.exists("/dls/science/users/iat69393/ptycho-tomo-project/coords_NEW2.csv"):
        #     keys = vals_to_save[0].keys()
            
        #     with open('coords_NEW2.csv', 'w', newline='') as output_file:
        #         dict_writer = csv.DictWriter(output_file, keys)
        #         dict_writer.writeheader()
        #         dict_writer.writerows(vals_to_save)

        self._proj_geom = astra.create_proj_geom('parallel3d_vec',  self._fsh[0], self._fsh[1], self._vec)
        return self._proj_geom


    def _update_data_at_id_astra(self, matrix, id):
        astra.data3d.store(id, matrix)

    def _delete_data_at_id_astra(self, id):
        astra.data3d.delete(id)


    def forward(self, input_vol, type = "FP3D_CUDA", iter=10, plot_one_view=False):

        # This is done here because it depends on the pods that are 
        # active (mpi) at the time this function is called
        self._create_proj_geometry()
        self._create_proj_array_and_ids()

        self._update_data_at_id_astra(np.real(input_vol), self._vol_id_real)
        self._update_data_at_id_astra(np.imag(input_vol), self._vol_id_imag)

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

        output_array = []
        for i, (k,v) in enumerate([(i,v) for i,v in self._obj.views.items() if v.pod.active]):
            real_part = _ob_views_real[i] 
            imag_part = _ob_views_imag[i]
            _obj = real_part + 1j * imag_part
            output_array.append(_obj)
        
        # Delete these as we don't need them any more
        self._delete_data_at_id_astra(self._proj_id_real)
        self._delete_data_at_id_astra(self._proj_id_imag)
        return output_array


    def backward(self, input_proj, type="BP3D_CUDA", iter=10):

        # This is done here because it depends on the pods that are 
        # active (mpi) at the time this function is called
        self._create_proj_geometry()
        self._create_proj_array_and_ids()
        
        self._update_data_at_id_astra(np.real(input_proj), self._proj_id_real)
        self._update_data_at_id_astra(np.imag(input_proj), self._proj_id_imag)

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
        
        # Delete these as we don't need them any more
        self._delete_data_at_id_astra(self._proj_id_real)
        self._delete_data_at_id_astra(self._proj_id_imag)
        
        return astra.data3d.get(self._vol_id_real) + 1j * astra.data3d.get(self._vol_id_imag)

