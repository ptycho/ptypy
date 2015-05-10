#! /usr/bin/python

import sys
from ptypy import io
from ptypy import utils as u

fname = sys.argv[1]
pars = u.Param()
pars.interactive=False

header = io.h5read(filename,'header')['header']
if str(header['kind']) == 'fullflat':
    raise NotImplementedError('Loading specific data from flattened dump not yet supported')
else: 
    content = io.h5read(filename,'content')['content']
    probes = content['probe']
    objects = content['object']
    runtime = content['runtime']

Plotter = u.MPLplotter(pars=pars, probes = probes, objects= objects, runtime= runtime)
Plotter.plot_all()
Plotter.draw()
