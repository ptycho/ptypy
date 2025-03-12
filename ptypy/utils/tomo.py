# -*- coding: utf-8 -*-
"""
Numerical util functions.

This file is part of the PTYPY package.

    :copyright: Copyright 2024 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
import scipy.ndimage as ndimage
import astra
import matplotlib.pyplot as plt
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


class AstraViewBased:

    """
    Base class for wrappers for the Astra projectors.
    """    

    def __init__(self, vol, n_views, view_shape, block_size, angles, shifts, view_to_proj_vectors):

        """
        Receives: 
        vol                     3D array representing the volume
        n_views                 number of views overall, across all blocks
        view_shape              shape of view
        block_size              size of chunk (a sub-part or equal to n_views)
        angles                  1D array of angles (same length as n_views)
        shifts                  2D array of x and y shifts, having shape: n_views x 2 (with 2 being x and y)
        view_to_proj_vectors    2D array of vectors, computed as differences between the center of views and 
                                the center of projections (same length as n_views)

        Does the fllowing:
         - the astra geometry for the projection array and the volume
         - the astra ids for real and imag parts of both proj array and volume.
         - sets up the configuration for both backward and forward 
        """

        self._n_views = n_views
        self._view_shape = view_shape
        self._block_size = block_size
        self._angles = angles
        self._shifts = shifts
        self._view_to_proj_vectors = view_to_proj_vectors
        self._ind_of_views = np.arange(self._block_size)
        
        self._init_proj_array_to_empty()   # Necessary for create_geom to work
        self._proj_geom = None
        self._proj_id_real = None
        self._proj_id_imag = None

        self._vol = vol   # Necessary for create_geom to work
        self._vol_geom = None
        self._vol_id_real = None
        self._vol_id_imag = None

        self._create_vector_for_proj_geom()
        self._create_proj_geom()
        self._create_proj_array_ids()

        self._create_vol_geom()
        self._create_vol_ids()

        # Setup config
        self._setup_config_for_forward()
        self._setup_config_for_backward()

    def _init_proj_array_to_empty(self):
        """
        Initialise proj_array to be empty
        """
        empty_proj_array = np.zeros(
            (self._block_size, self._view_shape[0], self._view_shape[1]), 
            dtype=np.complex64
            )
        self._proj_array = np.moveaxis(empty_proj_array, 1, 0)

    @property
    def proj_array(self):
        """
        Returns proj_array.
        """
        return self._proj_array

    @proj_array.setter
    def proj_array(self, proj_array):
        """
        Sets _proj_array to the value passed as input. 
        """ 
        # When this gets called, it's important to update ind_of_views too      
        self._proj_array = proj_array
        self._update_data_at_id_astra(self._proj_id_real, proj_array.real)
        self._update_data_at_id_astra(self._proj_id_imag, proj_array.imag)

    @property
    def ind_of_views(self):
        """
        Returns indexes of views in the block. 
        Has length of "proj_array" and so of "block_size".
        """
        return self._ind_of_views

    @ind_of_views.setter
    def ind_of_views(self, indeces):
        """
        Sets ind_of_views to the value passed as input. 
        """   
        self._ind_of_views = indeces

        # Create and update astra geometry
        self._create_proj_geom()
        self._update_geom_at_id_astra(self._proj_id_real, self._proj_geom)      
        self._update_geom_at_id_astra(self._proj_id_imag, self._proj_geom)

    @property
    def vol(self):
        """
        Returns the volume.
        """
        return self._vol

    @vol.setter
    def vol(self, vol):
        """
        Sets the volume and updates data at astra id.
        """   
        self._vol = vol

        self._update_data_at_id_astra(self._vol_id_real, np.real(vol))
        self._update_data_at_id_astra(self._vol_id_imag, np.imag(vol))
        
    def _update_data_at_id_astra(self, id, data):
        """
        Helper function that updates data at a certain astra id.
        """
        astra.data3d.store(id, data)

    def _update_geom_at_id_astra(self, id, new_geom):
        """
        Helper function that updates geometry at a certain astra id.
        Receives an astra geometry, e.g. the one that "_create_proj_geom"
        stores in "self._proj_geom"
        """
        astra.data3d.change_geometry(id, new_geom)

    def _create_proj_array_ids(self):
        """
        Creates astra ids for proj_array (real and imag).
        _create_geom must be called before this. 
        This function is called once only, when the class is initialised.        
        """
        self._proj_id_real = astra.data3d.create(
            "-sino", self._proj_geom, self._proj_array.real
            )
        self._proj_id_imag = astra.data3d.create(
            "-sino", self._proj_geom, self._proj_array.imag
            )

    def _create_vol_ids(self):
        """
        Creates astra ids for volume (real and imag).
        _create_vol_geom must be called before this. 
        This function is called once only, when the class is initialised.        
        """
        self._vol_id_real = astra.data3d.create(
            "-vol", self._vol_geom, self._vol.real
            )
        self._vol_id_imag = astra.data3d.create(
            "-vol", self._vol_geom, self._vol.imag
            )

    def _create_vol_geom(self):
        """Creates volume geometry"""
        self._vol_geom = astra.create_vol_geom(
            self._vol.shape[0], self._vol.shape[1], self._vol.shape[2]
            )

    def _create_vector_for_proj_geom(self):
        """
        Creates vector needed to construct the astra proj array geometry.
        """
        self._vec = np.zeros((self._n_views,12)) 
        for i in range(self._n_views):

            y = self._view_to_proj_vectors[i, 0] 
            x = self._view_to_proj_vectors[i, 1] 
            alpha = self._angles[i]
  
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

    def _create_proj_geom(self):
        """Creates proj_array geometry"""
        self._sub_vec = self._vec[[self._ind_of_views]]
        self._proj_geom = astra.create_proj_geom(
            'parallel3d_vec',  
            self._view_shape[0], 
            self._view_shape[1], 
            np.squeeze(self._sub_vec)
        )

    def _setup_config_for_forward(self, type="FP3D_CUDA"):
        """
        Gets the type, e.g. FP3D_CUDA, and prepares cfg.
        """
        cfg = astra.astra_dict(type)
        cfg["VolumeDataId"] = self._vol_id_real
        cfg["ProjectionDataId"] = self._proj_id_real
        self.forward_alg_id_real = astra.algorithm.create(cfg)

        cfg = astra.astra_dict(type)
        cfg["VolumeDataId"] = self._vol_id_imag
        cfg["ProjectionDataId"] = self._proj_id_imag
        self.forward_alg_id_imag = astra.algorithm.create(cfg)
        
    def _setup_config_for_backward(self, type="BP3D_CUDA"):
        """
        Gets the type, e.g. FP3D_CUDA, and prepares cfg.
        """
        cfg = astra.astra_dict(type)
        cfg["ReconstructionDataId"] = self._vol_id_real
        cfg["ProjectionDataId"] = self._proj_id_real
        self.backward_alg_id_real = astra.algorithm.create(cfg)

        cfg = astra.astra_dict(type)
        cfg["ReconstructionDataId"] = self._vol_id_imag
        cfg["ProjectionDataId"] = self._proj_id_imag
        self.backward_alg_id_imag = astra.algorithm.create(cfg)

    def forward(self, iter=1, out=None):
        """
        Computes forward projection, based on self._vol (this must have 
        been defined). Places output in "out" if provided.
        """
        astra.algorithm.run(self.forward_alg_id_real, iter)
        astra.algorithm.run(self.forward_alg_id_imag, iter)

        _proj_data_real = astra.data3d.get(self._proj_id_real)
        _proj_data_imag = astra.data3d.get(self._proj_id_imag)

        _ob_views_real = np.moveaxis(_proj_data_real, 0, 1)
        _ob_views_imag = np.moveaxis(_proj_data_imag, 0, 1)
        _ob_views_complex = _ob_views_real + 1j * _ob_views_imag
        
        if isinstance(out, np.ndarray):
            out[:] = _ob_views_complex
        elif out is None:
            out = _ob_views_complex
        else:
            raise ValueError(
                "The parameter 'out' provided as input to forward "
                "should be of type np.ndarray."
                )
        return out

    def backward(self, iter=1, out=None):
        """
        Computes backward projection, based on _proj_array (this must have 
        been defined). Places output in "out" if provided.
        """
        astra.algorithm.run(self.backward_alg_id_real, iter)
        astra.algorithm.run(self.backward_alg_id_imag, iter)

        _vol = astra.data3d.get(self._vol_id_real) + 1j * astra.data3d.get(self._vol_id_imag)

        if isinstance(out, np.ndarray):
            out[:] = _vol
        elif out is None:
            out = _vol
        else:
            raise ValueError(
                "The parameter 'out' provided as input to forward "
                "should be of type np.ndarray."
                )
        return out

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
