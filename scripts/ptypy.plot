#! /usr/bin/python

import sys

usage = """\
Usage:
%s recon_or_dump_file.ptyr [template] [dump_file]

Shows on screen a summary plot of a ptycho reconstruction.
Choose one of the templates `weak`,`black_and_white`,`nearfield`, ...

If dump_file is specified, the plot is dumped instead (in a format
that depends on the extension - typically .png, .jpg, .pdf or .svg).
""" % sys.argv[0]

if len(sys.argv) not in [2,3,4]:
    print usage
    sys.exit(0)

from ptypy import io
from ptypy import utils as u

filename = sys.argv[1]
template = sys.argv[2] if len(sys.argv)>2 else None

pars = u.Param()
pars.interactive=True

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
Plotter._set_autolayout(template)
Plotter.update_plot_layout()
Plotter.plot_all()
Plotter.draw()
Plotter.plot_all()
Plotter.draw()

def handle_close(evt):
    sys.exit()
Plotter.plot_fig.canvas.mpl_connect('close_event', handle_close)
try:
    Plotter.plot_fig.savefig(sys.argv[3],dpi = 300)
except IndexError:
    Plotter.draw()
finally:
    while True:
        u.pause(0.1)
    
