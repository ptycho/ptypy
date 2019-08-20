#!/usr/bin/env python2

import sys
import argparse

parser = argparse.ArgumentParser(description='Create a plot from reconstruction save or autosave.')
parser.add_argument('ptyrfile', type=str, help='path to *.ptyr compatible file')
parser.add_argument('-l', dest='layout', type=str, default='default',
                   help="""
                        layout of the plotter, use
                         - `default` for default
                         - `weak` for weakly scattering samples
                         - `black_and_white` for a less colourful appearance
                         - `nearfield` for full frame (useful for extended probe)
                        """)
parser.add_argument('-t', dest='imfile', type=str, 
                   help='image dump path, if specified the plot is saved instead \
                   (in a format that depends on the extension -\
                    typically .png, .jpg, .pdf or .svg)')
args = parser.parse_args()

from ptypy import io
from ptypy import utils as u

filename = args.ptyrfile
template = args.layout
save = args.imfile

pars = u.Param()
pars.interactive=True if save is None else False

header = io.h5read(filename,'header')['header']
if str(header['kind']) == 'fullflat':
    raise NotImplementedError('Loading specific data from flattened dump not yet supported')
else: 
    content = next(iter(io.h5read(filename,'content').values()))
    runtime = content['runtime']
    probes = u.Param()
    probes.update(content['probe'], Convert = True)
    objects = u.Param()
    objects.update(content['obj'], Convert = True)

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

#def handle_resize(evt):
#    Plotter.plot_all()
#Plotter.plot_fig.canvas.mpl_connect('resize_event', handle_resize)

if save is not None:
    Plotter.plot_fig.savefig(save,dpi = 300)
else:
    while Plotter:
        #Plotter.plot_all()
        Plotter.draw()
        u.pause(0.1)
    
usage = """\
Usage:
%s recon_or_dump_file.ptyr [template] [dump_file]

Shows on screen a summary plot of a ptycho reconstruction.
Choose one of the templates `weak`,`black_and_white`,`nearfield`, ...

If dump_file is specified, the plot is dumped instead (in a format
that depends on the extension - typically .png, .jpg, .pdf or .svg).
""" % sys.argv[0]
"""
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
"""
