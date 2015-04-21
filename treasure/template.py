# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 2013

@author: benders
"""
################ LOAD MODULES ##########################################
from pyE17 import utils as U
import ptypy
from ptypy import utils as u
import sys
import numpy as np
import os
########################################################################
p = u.Param()
##### GENERAL ####################################################
#plt.interactive(True)
#verbose.set_level(3)
p.base_path =  '/data/CDI/simulations/test'
##### PARAMETER INPUT ##################################################
# Please also define here what additional parameter means what
# As a default all parameters will form the simulation ID 
# First parameter is ALWAYS understood as stack index and will not be part of an id.
pyfile = os.path.basename(sys.argv[0])
pars=[]
if len(sys.argv)>1:
    pars += sys.argv[1:] 
    index_string = pars.pop(0)
"""
Use pars[ind] to pass changes through the script
Example:
dr = float(pars[0])
>> python simuscript.py <index> 1e-8
"""
### IO ###############################################################    
p.io=u.Param()
    # use this as base path for the simulation projects
    # the simulation will generate a folder in this path with 
    # the simulation id as folder name
p.io.base_path =  base_path
p.io.pyfile = pyfile
p.io.parstring = '_'.join(pars)                     # all other inputs except the first, which is considered a running scan number
p.io.experiment = ''.join(pyfile.split('.')[:-1])   # derived from name of this script or choose a custom name 
p.io.index_string = index_string                    # 1st argument, should act like scan number. TODO:accept everything that range() does
p.io.index_range = U.str2range(index_string)        # override index range formation from string here
p.io.savedirpattern = '%(base_path)s/%(experiment)s/%(parstring)s/analysis/'
p.io.dumpdirpattern = '%(base_path)s/%(experiment)s/%(parstring)s/dumps/'
p.io.recondirpattern = '%(base_path)s/%(experiment)s/%(parstring)s/recons/'
p.io.pythondirpattern = '%(base_path)s/%(experiment)s/%(parstring)s/python/'
##### PHYSICAL PARAMETERS ##############################################
p.geometry=u.Param()
    # define phyical constants of your experiment here

p.geometry.energy = 7.2            # Energy (in keV)
p.geometry.lam = None              # Wavelength in
p.geometry.z = 2.19                # Distance from object to screen 
p.geometry.d1 = 172e-6             # Pixel size in Detector plane
p.geometry.d0 = None               # Pixel sixe in Sample plane  
p.geometry.N = 220                 # Number of detector pixels
p.geometry.rebin = 1               # Rebinning
p.geometry.prop_type = 'farfield'  # propagation type 
p.geometry.antialiasing = 2         # use antialiasing when generating diffraction 

##### SCAN PARAMETERS ##############################################
p.scan = u.Param()            # Dictionary storing scan parameters
    #### Paramaters for popular scan methods 
p.scan.scan_type = 'round_roi' # ['round', 'raster', 'round_roi','custom']
p.scan.dr = 0.3e-6            # round,round_roi :width of shell 
p.scan.nr = 10                 # round : number of intervals (# of shells - 1) 
p.scan.nth = 5                # round,round_roi: number of points in the first shell 
p.scan.lx = 6e-6              # round_roi: Width of ROI 
p.scan.ly = 3e-6              # round_roi: Height of ROI 
p.scan.nx = 10                 # raster scan: number of steps in x
p.scan.ny = 10                 # raster scan: number of steps in y
p.scan.dx = 1e-6               # raster scan: step size (grid spacing)
p.scan.dy = 1e-6               # raster scan: step size (grid spacing)
    #### other 
p.scan.positions=[]            # fill this list with your own script if you want other scan patterns
p.scan.noise = 1e-10           # possible position inaccuracy as noise on coordinates

##### PROBE PARAMETERS #################################################
p.illumination=u.Param()
#### define your illumination here #############
p.illumination.probe_type = 'parallel'		# 'focus','parallel','path_to_file'
p.illumination.aperture_type = 'circ'			# 'rect','circ','path_to_file'
p.illumination.aperture_size = None           # aperture diameter
p.illumination.aperture_edgewidth = 1         # edge smoothing width of aperture in pixel        
p.illumination.focal_dist = 0.1               # distance from prefocus aperture to focus
p.illumination.prop_dist = 0.001              # propagation distance from focus (or from aperture if parallel)
p.illumination.UseConjugate = False           # use the conjugate of the probe instef of the probe
p.illumination.antialiasing = 4.0             # antialiasing factor used when generating the probe
p.illumination.diameter = 500e-9              # if aperture_size = None this parameter is used instead. 
                                            # Gives the desired probe size in sample plane
p.illumination.photons = 1e8                  # photons in the probe
##### OBJECT PARAMETERS ################################################
p.sample=u.Param()
#expects a 3d-numpy array which is a stack of projections, first Axis is projection no
p.sample.objectfile ='/data/CDI/simulations/Objects/projections_for_bjoern_7_2keV.npy'
    
p.sample.offset = (0,0)    # offset= offset_list(int(par[0]))       
                          # (offsetx,offsety) move scan pattern relative to center in pixel 

p.sample.zoom = 3          # None, scalar or 2-tupel. If None, the pixel is assumed to be right 
                          # otherwise the image will be resized using ndimage.zoom            
    
p.sample.ref_index = None  # If None, treat projection as projection of refractive index/
                          # If a refractive index is provided the object's absolut value will be
                          # used to scale the refractive index.              
    
p.sample.smoothing_mfs = None # Smooth with minimum feature size (in pixel units) if not None
    
p.sample.fill = 1.0        # if object is smaller than the objectframe, fill with fill: 





