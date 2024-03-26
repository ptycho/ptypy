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


def _weights(x, dx=1, orig=0):
    x = np.ravel(x)
    floor_x = np.floor((x - orig) / dx).astype(np.int64)
    alpha = (x - orig - floor_x * dx) / dx
    return np.hstack((floor_x, floor_x + 1)), np.hstack((1 - alpha, alpha))

def _generate_center_coordinates(l_x):
    X, Y = np.mgrid[:l_x, :l_x].astype(np.float64)
    center = l_x / 2.0
    X += 0.5 - center
    Y += 0.5 - center
    return X, Y

## This code is based on this example: https://scikit-learn.org/stable/auto_examples/applications/plot_tomography_l1_reconstruction.html
def forward_projector_matrix_tomo(size, nangles):
    X, Y = _generate_center_coordinates(size)
    angles = np.linspace(0, np.pi, nangles, endpoint=False)
    data_inds, weights, camera_inds = [], [], []
    data_unravel_indices = np.arange(size**2)
    data_unravel_indices = np.hstack((data_unravel_indices, data_unravel_indices))
    for i, angle in enumerate(angles):
        Xrot = np.cos(angle) * X - np.sin(angle) * Y
        inds, w = _weights(Xrot, dx=1, orig=X.min())
        mask = np.logical_and(inds >= 0, inds < size)
        for j in range(size):
            weights += list(w[mask])
            camera_inds += list(inds[mask] + j * size + i * size * size)
            data_inds += list(data_unravel_indices[mask] + j * size * size)
    return sparse.coo_matrix((weights, (camera_inds, data_inds)))

## This code is based on this example: https://scikit-learn.org/stable/auto_examples/applications/plot_tomography_l1_reconstruction.html
def forward_projector_matrix_ptychotomo(vsize, nangles, fsize, pos):
    npos = pos.shape[1]
    X, Y = _generate_center_coordinates(vsize)
    angles = np.linspace(0, np.pi, nangles, endpoint=False)
    data_inds, weights, camera_inds = [], [], []
    data_unravel_indices = np.arange(vsize**2)
    data_unravel_indices = np.hstack((data_unravel_indices, data_unravel_indices))
    for i, angle in enumerate(angles):
        Xrot = np.cos(angle) * X - np.sin(angle) * Y
        inds, w = _weights(Xrot, dx=1, orig=X.min())
        mask = np.logical_and(inds >= 0, inds < vsize)
        for k in range(npos):
            py,px = pos[:,k]
            pmask = np.logical_and(inds >= px, inds < px+fsize)
            for j in np.arange(py, py+fsize, 1): 
                weights += list(w[mask&pmask])
                camera_inds += list(inds[mask&pmask]-px + (j-py) * fsize + k * fsize * fsize + i * npos * fsize * fsize)
                data_inds += list(data_unravel_indices[mask&pmask] + j * vsize * vsize)

    camera_inds = [0 if i < 0 else int(i) for i in camera_inds]
    data_inds = [0 if i < 0 else int(i) for i in data_inds]
    return sparse.coo_matrix((weights, (camera_inds, data_inds)))

def sirt_projectors1(A,b):
    R = sparse.lil_matrix((A.shape[0], A.shape[0]))
    R.setdiag(np.array(1./A.T.sum(axis=0))[0])
    C = sparse.lil_matrix((A.shape[1], A.shape[1]))
    C.setdiag(np.array(1./A.sum(axis=0))[0])
    return C@A.T@R@b, C@A.T@R@A

def sirt_projectors2(A):
    print('Starting sirt_projectors2')
    R = sparse.lil_matrix((A.shape[0], A.shape[0]))
    R.setdiag(np.array(1./A.T.sum(axis=0))[0])
    C = sparse.lil_matrix((A.shape[1], A.shape[1]))
    C.setdiag(np.array(1./A.sum(axis=0))[0])
    print('Finishing sirt_projectors2')
    return C@A.T@R, C@A.T@R@A

def sample_volume(N):
    vol = np.zeros((N,N,N))
    xx,yy,zz = np.meshgrid(np.arange(N)-N/2, np.arange(N)-N/2,np.arange(N)-N/2)
    m = lambda r,dx,dy,dz: np.sqrt((xx+dx)**2 + (yy+dy)**2 + (zz+dz)**2) < r
    vol[m(N//6,0,0,0)] = 1
    vol[m(N//8,N//7,N//7,N//7)] = 2
    vol[m(N//12,N//12,0,0)] = 5
    vol[m(N//12,0,N//6,0)] = 5
    return ndimage.gaussian_filter(vol,1)

def refractive_index_map(Nx):
    beta = np.log(sample_volume(Nx)+20)-np.log(20)
    delta = 0.1 * sample_volume(Nx)
    return delta + 1j * beta