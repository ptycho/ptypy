#!/usr/bin/env python3

import argparse
from ptypy import defaults_tree
from ptypy.utils.plot_client import MPLClient

plot_desc = defaults_tree['ptycho.io.autoplot']
client_desc = defaults_tree['io.interaction']

def ptypy_plotclient():
    opt = parse()
    launch_plotclient(opt)

def parse():
    parser = argparse.ArgumentParser(description='Create a ZeroMQ plotting client.')
    excludes = ['active']
    parser = plot_desc.add2argparser(parser, excludes=excludes)
    excludes = ['active', 'threaded', 'connections', 'interval']
    parser = client_desc.add2argparser(parser, excludes=excludes)
    parser.add_argument('-d', '--dir', dest='directory', type=str, default='./',
                        help='image dump directory')

    # Parse command line arguments. This will update defaults in the defaults_tree
    args = parser.parse_args()
    return args

def launch_plotclient(args):
    plot_pars = plot_desc.make_default(depth=5)
    client_pars = client_desc.make_default(depth=5)

    plotter = MPLClient(client_pars=client_pars, autoplot_pars=plot_pars)
    plotter.loop_plot()

if __name__ == "__main__":
    ptypy_plotclient()
