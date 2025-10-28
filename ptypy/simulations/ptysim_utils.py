# -*- coding: utf-8 -*-
"""
Utilities for the simulation package.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
import os
import time
from scipy import ndimage


from ptypy import utils as u
#from ptypy.core import data

__all__ = ['exp_positions']

def czoom(input,*args,**kwargs):
    return ndimage.zoom(input.real, *args, **kwargs) + 1j*ndimage.zoom(input.imag, *args, **kwargs)

def exp_positions(positions, drift=0.0,scale= 0.0,noise=0.0):
    """
    Takes postion array and messes around with it
    """
    pos = np.array(positions)
    offset = augment_to_coordlist(drift,len(pos))
    scale = augment_to_coordlist(scale,len(pos))
    point_distance = u.norm(pos[1]-pos[0])

    if noise is not None:
        pos += offset + scale*np.random.normal(0,noise*point_distance,pos.shape)
    else:
        pos += np.zeros(pos.shape)

    return pos

""" def make_sim_datasource(model_inst,drift=0.0,scale= 0.0,noise=0.0):

    labels=[]
    sources =[]
    pars =[]
    for label,scan in model_inst.scans.items():
        source = scan.pars.source
        if source is None:
            source = model_inst.ptycho.paths.get_data_file(label=label)
        labels.append(label)
        sources.append(None)
        #p = u.Param()
        scan_info=u.Param()
        pos = scan.pos_theory
        scan_info.positions_theory = pos
        scan_info.positions = exp_positions(pos,drift,scale,noise)
        N = u.expect2(scan.pars.geometry.N)
        scan_info.shape = (len(pos),np.int(N[0]),np.int(N[1]))
        scan_info.scan_label = label
        scan_info.data_filename = source

        pars.append(scan_info)

    return data.StaticDataSource(sources,pars,labels)



def framepositions(pos_pixel,probe_shape,frame_overhead=(10,10)):
    pos_pixel -= pos_pixel.min(axis=0)
    pos_pixel = np.round(pos_pixel)
    shape = pos_pixel.max(axis=0)+w.expect2(probe_shape)+w.expect2(frame_overhead)
    pos_pixel += np.round(w.expect2(frame_overhead) /2)
    positions = [(p[0],p[0]+probe_shape[0],p[1],p[1]+probe_shape[1]) for p in pos_pixel]

    return shape,positions """


def augment_to_coordlist(a,Npos):
    if np.isscalar(a):
        a=u.expect2(a)

    a = np.asarray(a)
    if a.size % 2 == 0:
        a=a.reshape(a.size//2,2)

    if a.shape[0] < Npos:
        b=np.concatenate((1+Npos//a.shape[0])*[a],axis=0)
    else:
        b=a

    return b[:Npos,:2]
