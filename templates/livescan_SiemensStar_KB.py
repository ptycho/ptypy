"""
Live-reconstruction for NanoMAX ptycho.

"""

import os
import sys
import time
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
from distutils.version import LooseVersion
from mpi4py import MPI
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
setting='cpu'
if ptypy.version[:3] == '0.5':
	ptypy.load_ptyscan_module("livescan")
	if setting != 'cpu':
		ptypy.load_gpu_engines(arch="cuda")
		# ptypy.load_gpu_engines(arch="cuda"):::'DM_pycuda', 'DM_pycuda_nostream',  ptypy.load_gpu_engines(arch="serial"):::'DM_serial',  ptypy.load_gpu_engines(arch="ocl"):::'DM_ocl'

print(ptypy.__file__)
############################################################################
# hard coded user input
############################################################################

detector         = 'eiger' # or 'merlin' or 'pilatus'
##### beamtime_basedir = '/home/reblex/Documents/Data/' ##'/data/visitors/nanomax/20200116/2021112508'
sample           = 'nanomax_siemens_KB'
scannr           = 6
distance_m       = 3.65	   # distance between the sample and the detector in meters
defocus_um       = 900	   # distance between the focus and the sample plane in micro meters -> used for inital probe
#energy_keV      = 6.5	   # incident photon energy in keV ... now read from scan file


if len(sys.argv)>=2:
	# scan number is given as first argument of this script
	scannr = int(sys.argv[1])

if len(sys.argv)>=3:
	# ..then this is running with slurm
	# sample name (and thus data directory) is given as the 2nd argument
	slurmjobname = str(sys.argv[2])
else:
	slurmjobname = 'tmp'

if len(sys.argv)>=4:
	# ..then this is running with slurm
	# slurm job-id is given as the 3rd argument, added to the name of copied .sh script
	jobid = str(sys.argv[3])
else:
	jobid = ''

############################################################################
# some preparations before the actual reconstruction
############################################################################

out_dir			= f'/home/reblex/Documents/Reconstructions/{sample}_{scannr:06d}_{slurmjobname}/'
out_dir_data    = out_dir + 'data/'
out_dir_dumps   = out_dir + 'dumps/'
out_dir_scripts = out_dir + 'scripts/'
out_dir_rec     = out_dir + 'rec/'

# and what the files are supposed to be called
path_data       = out_dir_data  + 'data_scan_' + str(scannr).zfill(6) + '.ptyd'							    # the file with the prepared data
path_dumps      = out_dir_dumps + 'dump_scan_' + str(scannr).zfill(6)+'_%(engine)s_%(iterations)04d.ptyr'   # intermediate results
path_rec        = out_dir_rec   + 'rec_scan_' + str(scannr).zfill(6)+'_%(engine)s_%(iterations)04d.ptyr'	# final reconstructions (of each engine)

# stuff to only do once
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank==0:

	# create output directories if it does not already exists
	os.makedirs(out_dir,         exist_ok=True)
	os.makedirs(out_dir_data,    exist_ok=True)
	os.makedirs(out_dir_dumps,   exist_ok=True)
	os.makedirs(out_dir_scripts, exist_ok=True)
	os.makedirs(out_dir_rec,     exist_ok=True)

	# copy this file into this directory with a tag made from the time and date this script was run
	os.system('cp ' + str(__file__) + ' ' + out_dir_scripts + time.strftime("%Y-%m-%d_%H%M") + '_' + str(__file__).split('/')[-1])  ##
	if len(jobid) > 0:
		os.system('cp /home/reblex/Documents/Scripts/slurm_submit_Lex.sh' + ' ' + out_dir_scripts + f'slurm_submit_Lex_{slurmjobname}_{jobid}'.rstrip('_') + '.sh')  ##


############################################################################
# creating the parameter tree
############################################################################

# General parameters
p = u.Param()
p.verbose_level = 3#'interactive'  # 3
p.run = 'scan%d' % scannr

# where to put the reconstructions
p.io = u.Param()
p.io.home = out_dir_rec                     # where to save the final reconstructions
p.io.rfile = path_rec                       # how to name those files for the final reconstructions
p.io.interaction = u.Param()
p.io.interaction.active = True
p.io.autosave = u.Param()
p.io.autosave.rfile = path_dumps            # where to save the intermediate reconstructions and how to name them
p.io.autoplot = u.Param()
p.io.autoplot.active = True



# Scan parameters
p.scans = u.Param()
p.scans.scan00 = u.Param()
p.scans.scan00.name = 'Full'
p.scans.scan00.coherence = u.Param()
p.scans.scan00.coherence.num_probe_modes = 1		# number of probe modes
p.scans.scan00.coherence.num_object_modes = 1		# number of object modes


p.scans.scan00.data = u.Param()
p.scans.scan00.data.name = 'LiveScan'
p.scans.scan00.data.block_wait_count = 1
##### p.scans.scan00.data.path = beamtime_basedir+sample+'/'
p.scans.scan00.data.detector = detector
##### p.scans.scan00.data.maskfile = {'merlin': '/data/visitors/nanomax/common/masks/merlin/latest.h5', 'pilatus': None, 'eiger': None,}[detector]
##### p.scans.scan00.data.scanNumber = scannr
p.scans.scan00.data.xMotor = 'pseudo/x'
p.scans.scan00.data.yMotor = 'basey'
p.scans.scan00.data.relay_host = 'tcp://127.0.0.1'
p.scans.scan00.data.relay_port = 45678
##### p.scans.scan00.data.zDetectorAngle = 0.0    # rotation of the detector around the beam axis in [deg]
p.scans.scan00.data.shape = 256				# size of the window of the diffraction patterns to be used in pixel
p.scans.scan00.data.save = 'append'
p.scans.scan00.data.dfile = path_data		# once all data is collected, save it as .ptyd file
p.scans.scan00.data.center = (1340, 646)     # center of the diffraction pattern (y,x) in pixel or None -> auto
##### p.scans.scan00.data.cropOnLoad = True       # only load used part of detector frames -> save memory
                                            # requires center to be set explicitly
p.scans.scan00.data.xMotorFlipped = False
p.scans.scan00.data.yMotorFlipped = False
p.scans.scan00.data.orientation = {'merlin': (False, False, True),
                                   'pilatus': None,
                                    'eiger': (False, True, False)}[detector]
p.scans.scan00.data.distance = distance_m   # distance between sample and detector in [m]
p.scans.scan00.data.psize = {'pilatus': 172e-6,
                             'merlin': 55e-6,
                              'eiger': 75e-6}[detector]
#p.scans.scan00.data.energy = energy_keV    # incident photon energy in [keV], now read from file
##### p.scans.scan00.data.I0 = None               # can be like 'alba2/1'
p.scans.scan00.data.min_frames = 1		## Minimum number of frames loaded by each node/process
p.scans.scan00.data.start_frame = 15		## Minimum number of frames loaded before starting iterations
p.scans.scan00.data.frames_per_iter = None  # None ## Load a fixed number of frames in between each iteration, default = None
p.scans.scan00.data.load_parallel = 'all'

# scan parameters: illumination
p.scans.scan00.illumination = u.Param()

p.scans.scan00.illumination.model = None                              # option 1: probe is initialized from a guess
p.scans.scan00.illumination.aperture = u.Param()
p.scans.scan00.illumination.aperture.form = 'rect'                    # initial probe is a rectangle (KB focus)
p.scans.scan00.illumination.aperture.size = 200e-9			          # of this size in [m] the focus
p.scans.scan00.illumination.propagation = u.Param()
p.scans.scan00.illumination.propagation.parallel = 1.*defocus_um*1e-6 # propagate the inital guess -> gives phase curvature

#p.scans.scan00.illumination.model = 'recon'                           # option 2: probe is initialized from a previous reconstruction
#p.scans.scan00.illumination.recon = u.Param()
#p.scans.scan00.illumination.recon.rfile = ...                         # absolute path to a .ptyr file containing the probe to be used as initial guess
#p.scans.scan00.illumination.aperture = u.Param()
#p.scans.scan00.illumination.aperture.form = 'rect'                    # this aperture is not optional
#p.scans.scan00.illumination.aperture.size = 10e-8                     # either make it very large, or cut down the loaded probe




############################################################################
# 1st use the difference map algorithm
############################################################################

# general
p.engines = u.Param()
p.engines.engine00 = u.Param()
if setting == 'cpu':
	p.engines.engine00.name = 'DM'
else:
	p.engines.engine00.name = 'DM_pycuda'#'DM'
p.engines.engine00.numiter = 50                    # number of iterations
p.engines.engine00.numiter_contiguous = 1##50          # Number of iterations without interruption

p.engines.engine00.probe_support = 3                # non-zero probe area as fraction of the probe frame
#p.engines.engine00.probe_update_start = 50          # number of iterations before probe update starts
#p.scans.scan00.illumination.model = 'recon'         # To bring previous reconst model
#p.scans.scan00.illumination.recon = u.Param()
#p.scans.scan00.illumination.recon.rfile = probe_previous
#p.scans.scan00.illumination.aperture = u.Param()
#p.scans.scan00.illumination.aperture.form = 'circ'
#p.scans.scan00.illumination.aperture.size = 8000e-9
#p.scans.scan00.illumination.propagation = u.Param()
############################################################################
# 2nd use the maximum likelyhood algorithm
############################################################################
#
# # general
# p.engines.engine01 = u.Param()
# if setting == 'cpu':
# 	p.engines.engine01.name = 'ML'
# else:
# 	p.engines.engine01.name = 'ML_pycuda'#'ML'
# p.engines.engine01.numiter = 10 #20                    # number of iterations
# p.engines.engine01.numiter_contiguous = 10##50          # save a dump file every x iterations
############################################################################
# start the reconstruction
############################################################################

if LooseVersion(ptypy.version) < LooseVersion('0.3.0'):
    raise Exception('Use ptypy 0.3.0 or better!')

#p.frames_per_block = 3 #100 #400 ####default: 100000
t0 = time.time()
def runptycho():
	print('TIME 0: ', time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()))
	P = Ptycho(p, level=1)
	print('TIME 1: ', time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()))
	print('Ptycho init level 2')
	P.init_data()
	print('TIME 2: ', time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()))
	print('Ptycho init level 3')
	P.init_communication()
	print('TIME 3: ', time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()))
	print('Ptycho init level 4')
	P.init_engine()
	print('TIME 4: ', time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()))
	print('Ptycho init level 5')
	t1 = time.time()
	P.run()
	print('TIME 5a: ', time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()))
	P.finalize()
	print('TIME 5b: ', time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()))
	return P, t1
# P, t1 = runptycho()
# print(f'\nP1-P4 took {t1-t0} seconds, P5 took {time.time()-t1} seconds.')

print('TIME START: ', time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()))
P = Ptycho(p, level=5)
print('TIME END: ', time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()))
print(f'\nEverything took {time.time()-t0} seconds.')

