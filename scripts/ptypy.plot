#! /usr/bin/python

import sys

usage = """\
Usage:
%s recon_or_dump_file.ptyr [dump_file]

Shows on screen a summary plot of a ptycho reconstruction.
If dump_file is specified, the plot is dumped instead (in a format
that depends on the extension - typically .png, .jpg, .pdf or .svg).
""" % sys.argv[0]

if len(sys.argv) not in [2,3]:
    print usage
    sys.exit(0)

from ptypy import io
from ptypy import utils as u

filename = sys.argv[1]
pars = u.Param()
pars.interactive=False

header = io.h5read(filename,'header')['header']
if str(header['kind']) == 'fullflat':
    raise NotImplementedError('Loading specific data from flattened dump not yet supported')
else: 
    content = io.h5read(filename,'content')['content']
    probes = u.Param()
    probes.update(content['probe'], Convert = True)
    objects = u.Param()
    objects.update(content['obj'], Convert = True)
    runtime = content['runtime']

Plotter = u.MPLplotter(pars=pars, probes = probes, objects= objects, runtime= runtime)
Plotter.update_plot_layout()
Plotter.plot_all()
try:
    Plotter.plot_fig.savefig(sys.argv[2],dpi = 300)
except IndexError:
    Plotter.draw()
    while True:
        u.pause(0.1)
    
