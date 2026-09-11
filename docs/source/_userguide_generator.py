import os
import numpy as np
import matplotlib.pyplot as plt
from ptypy import utils as u
from ptypy.core import geometry, illumination, Storage, Container, Ptycho

# if output directory does not exist, create it
if not os.path.exists("./userguide/generated/"):
	os.mkdir("./userguide/generated/")



def create_test_image(outdir="./userguide/generated/", outfile="test.png"):
	data = np.random.random((20,2))
	plt.figure()
	plt.plot(data[:,0], data[:,1], c='r', marker='>')
	plt.tight_layout()
	plt.savefig(f'{outdir}{outfile}')


def create_all_init_probe_figures(outdir="./userguide/generated/"):
	create_one_init_probe_figure(p=make_probe_example_01(), outdir=outdir, outfile="init_probe_example_01.png")
	create_one_init_probe_figure(p=make_probe_example_02(), outdir=outdir, outfile="init_probe_example_02.png")
	#create_one_init_probe_figure(storage=make_probe_example_11(), outdir=outdir, outfile="init_probe_example_11.png")
	create_one_init_probe_figure(p=make_probe_example_21(), outdir=outdir, outfile="init_probe_example_21.png")
	create_one_init_probe_figure(p=make_probe_example_22(), outdir=outdir, outfile="init_probe_example_22.png")
	create_one_init_probe_figure(p=make_probe_example_23(), outdir=outdir, outfile="init_probe_example_23.png")
	create_one_init_probe_figure(p=make_probe_example_31(), outdir=outdir, outfile="init_probe_example_31.png")
	create_one_init_probe_figure(p=make_probe_example_32(), outdir=outdir, outfile="init_probe_example_32.png")
	create_one_init_probe_figure(p=make_probe_example_33(), outdir=outdir, outfile="init_probe_example_33.png")
	create_one_init_probe_figure(p=make_probe_example_41(), outdir=outdir, outfile="init_probe_example_41.png")
	create_one_init_probe_figure(p=make_probe_example_42(), outdir=outdir, outfile="init_probe_example_42.png")
	create_one_init_probe_figure(p=make_probe_example_43(), outdir=outdir, outfile="init_probe_example_43.png")
	create_one_init_probe_figure(p=make_probe_example_44(), outdir=outdir, outfile="init_probe_example_44.png")
	create_one_init_probe_figure(p=make_probe_example_51(), outdir=outdir, outfile="init_probe_example_51.png")
	create_one_init_probe_figure(p=make_probe_example_52(), outdir=outdir, outfile="init_probe_example_52.png")
	create_one_init_probe_figure(p=make_probe_example_53(), outdir=outdir, outfile="init_probe_example_53.png")
	create_one_init_probe_figure(p=make_probe_example_54(), outdir=outdir, outfile="init_probe_example_54.png")
	create_one_init_probe_figure(p=make_probe_example_61(), outdir=outdir, outfile="init_probe_example_61.png")
	create_one_init_probe_figure(p=make_probe_example_62(), outdir=outdir, outfile="init_probe_example_62.png")
	create_one_init_probe_figure(p=make_probe_example_63(), outdir=outdir, outfile="init_probe_example_63.png")



def create_one_init_probe_figure(p=None, outdir="./userguide/generated/", outfile="init_probe_test.png"):
	G, g = make_geometry()
	s = Storage(Container(ID='probe'), shape=(1, g.shape, g.shape), psize=G.resolution)
	illumination.init_storage(s, p, energy=g.energy, shape=(g.shape,g. shape))
	extent = [0, np.shape(s.data)[2] * s._psize[1] * 1.e6, 0,  np.shape(s.data)[1] * s._psize[0] * 1.e6]

	plt.figure(figsize=(8,3), dpi=100)

	plt.subplot(1,2,1)
	plt.title('initlial probe - amplitude')
	plt.imshow(np.abs(s.data[0]), interpolation='None', cmap='Greys_r', extent=extent)
	plt.colorbar()
	plt.xlabel('um')
	plt.ylabel('um')
	plt.subplot(1,2,2)
	plt.title('initlial probe - phase')
	plt.imshow(np.angle(s.data[0]), interpolation='None', cmap='hsv', extent=extent, vmin=-np.pi, vmax=np.pi)
	plt.colorbar()
	plt.xlabel('um')
	plt.ylabel('um')
	plt.tight_layout()
	plt.savefig(f'{outdir}{outfile}')

def make_geometry():
	g = u.Param()
	g.energy = 12.4    # photon energy in keV
	g.distance = 3.16  # detector distance in m
	g.psize = 75e-6    # detector pixel size in m
	g.shape = 128      # size of the probe array
	g.propagation = 'farfield'
	G = geometry.Geo(owner=None, pars=g)
	return G, g

def make_probe_example_01():
	probe = np.zeros((1, 256,256), dtype=complex) 
	probe[0, 64:128, 64:128] = 1. * np.exp(1.j * -0.5 * np.pi)
	probe[0, 128:192, 32:96] = 2. * np.exp(1.j * 0.5 * np.pi)
	p = u.Param()
	p.model = probe 
	p.aperture = u.Param()
	p.aperture.form = 'rect' 
	p.aperture.size = 10e-6
	return p

def make_probe_example_02():
	probe = np.zeros((1, 256,256), dtype=complex) 
	probe[0, 64:128, 64:128] = 1. * np.exp(1.j * -0.5 * np.pi)
	probe[0, 128:192, 32:96] = 2. * np.exp(1.j * 0.5 * np.pi)
	p = u.Param()
	p.model = probe 
	p.aperture = u.Param()
	return p

def make_probe_example_11():
	# ToDo: implenet example with loading a probe from a previous reconstruction
	pass

def make_probe_example_21():
	p = u.Param()
	p.model = None
	p.aperture = u.Param()
	p.aperture.size = 500e-9 # in meters
	return p

def make_probe_example_22():
	p = u.Param()
	p.model = None
	p.aperture = u.Param()
	p.aperture.size = 2000e-9 # in meters
	return p

def make_probe_example_23():
	p = u.Param()
	p.model = None
	p.aperture = u.Param()
	p.aperture.size = (2000e-9, 500e-9) 
	return p

def make_probe_example_31():
	p = u.Param()
	p.model = None
	p.aperture = u.Param()
	p.aperture.form = 'circ' # default
	p.aperture.size = 2000e-9
	return p

def make_probe_example_32():
	p = u.Param()
	p.model = None
	p.aperture = u.Param()
	p.aperture.form = 'rect'
	p.aperture.size = 2000e-9
	return p

def make_probe_example_33():
	p = u.Param()
	p.model = None
	p.aperture = u.Param()
	p.aperture.form = 'rect'
	p.aperture.size = (500e-9, 2000e-9)
	return p

def make_probe_example_41():
	p = u.Param()
	p.model = None
	p.aperture = u.Param()
	p.aperture.form = 'rect'
	p.aperture.size = (500e-9, 2000e-9)
	p.aperture.rotate = 0.15 * 3.1415 # angle in radians
	return p

def make_probe_example_42():
	p = u.Param()
	p.model = None
	p.aperture = u.Param()
	p.aperture.form = 'rect'
	p.aperture.size = 2000e-9
	p.aperture.central_stop = 0.20 
	return p

def make_probe_example_43():
	p = u.Param()
	p.model = None
	p.aperture = u.Param()
	p.aperture.form = 'rect'
	p.aperture.size = 2000e-9
	p.aperture.edge = 20 # in pixels
	return p

def make_probe_example_44():
	p = u.Param()
	p.model = None
	p.aperture = u.Param()
	p.aperture.form = 'rect'
	p.aperture.size = 2000e-9
	p.aperture.offset = (500e-9, 1000e-9) # in m
	return p

def make_probe_example_51():
	p = u.Param()
	p.model = None
	p.aperture = u.Param()
	p.aperture.form = 'rect'
	p.aperture.size = 2000e-9
	p.aperture.diffuser = (0.5 * 3.1415, 5) 
	return p

def make_probe_example_52():
	p = u.Param()
	p.model = None
	p.aperture = u.Param()
	p.aperture.form = 'rect'
	p.aperture.size = 2000e-9
	p.aperture.diffuser =  (1 * 3.1415, 2)
	return p

def make_probe_example_53():
	p = u.Param()
	p.model = None
	p.aperture = u.Param()
	p.aperture.form = 'rect'
	p.aperture.size = 2000e-9
	p.aperture.diffuser = (0 * 3.1415, 0 , 0.7, 5) 
	return p

def make_probe_example_54():
	p = u.Param()
	p.model = None
	p.aperture = u.Param()
	p.aperture.form = 'rect'
	p.aperture.size = 2000e-9
	p.aperture.diffuser = (0.5 * 3.1415, 10 , 0.7, 5) 
	return p

def make_probe_example_61():
	p = u.Param()
	p.model = None
	p.aperture = u.Param()
	p.aperture.form = 'circ'
	p.aperture.size = 1e-6
	p.propagation = u.Param()
	p.propagation.parallel = 1e-3 # distance in m
	return p

def make_probe_example_62():
	p = u.Param()
	p.model = None
	p.aperture = u.Param()
	p.aperture.form = 'rect'
	p.aperture.size = 525e-6   		# aperture diameter of the KB mirrors
	p.propagation = u.Param()
	p.propagation.focussed = 0.200  # focal length of the KB mirror(s)
	p.propagation.parallel = 500e-6 # distance sample to focus
	p.propagation.antialiasing = 1
	return p

def make_probe_example_63():
	p = u.Param()
	p.model = None
	p.aperture = u.Param()
	p.aperture.form = 'circ'
	p.aperture.size = 100e-6   		# aperture diameter of the FZP
	p.aperture.central_stop = 25e-6 / p.aperture.size
	p.propagation = u.Param()
	p.propagation.focussed = 0.18   # focal length of FZP
	#p.propagation.parallel = 100e-6 # distance sample to focus
	p.propagation.antialiasing = 1
	return p
