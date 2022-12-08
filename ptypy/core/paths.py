# -*- coding: utf-8 -*-
"""
Path manager.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import sys
import os

from .. import utils as u

__all__ = ['DEFAULT', 'Paths']

DEFAULT = u.Param(
    # (03) Relative base path for all other paths
    home="./",
    # (07) filename for dumping plots
    autoplot="plots/%(run)s/%(run)s_%(engine)s_%(iteration)04d.png",
    # (10) directory to save final reconstruction
    recon="recons/%(run)s/%(run)s_%(engine)s.ptyr",
    # (12) directory to save intermediate results runtime parameters
    autosave="dumps/%(run)s/%(run)s_%(engine)s_%(iteration)04d.ptyr",
)
""" Default path parameters. See :py:data:`.io.paths`
    and a short listing below """


class Paths(object):
    """
    Path managing class
    """
    DEFAULT = DEFAULT

    def __init__(self, io=None):
        """
        Parameters
        ----------
        io : Param or dict
            Parameter set to pick path info from. See :py:data:`.io`
        """
        self.runtime = u.Param(
            run=os.path.split(sys.argv[0])[1].split('.')[0],
            engine="None",
            iteration=0,
            iterations=0,
        )

        self.home = io.get('home', self.DEFAULT.home)

        try:
            self.autosave = io.autosave.rfile
        except:
            self.autosave = self.DEFAULT.autosave
        try:
            self.autoplot = io.autoplot.imfile
        except:
            self.autoplot = self.DEFAULT.autoplot
        try:
            self.recon = io.rfile
        except:
            self.recon = self.DEFAULT.recon

        sep = os.path.sep
        if not self.home.endswith(sep):
            self.home += sep

        for key in ['autosave', 'autoplot', 'recon']:
            v = self.__dict__[key]
            if isinstance(v, str):
                if not v.startswith(os.path.sep):
                    self.__dict__[key] = self.home + v

    def run(self, run):
        """
        Determine run name
        """
        return self.runtime.run if run is None else run

    def auto_file(self, runtime=None):
        """ File path for autosave file """
        return self.get_path(self.autosave, runtime)

    def recon_file(self, runtime=None):
        """ File path for reconstruction file """
        return self.get_path(self.recon, runtime)

    def plot_file(self, runtime=None):
        """
        File path for plot file
        """
        p = self.get_path(self.autoplot, runtime)
        return self.get_path(self.autoplot, runtime)

    def get_path(self, path, runtime):
        if runtime is not None:
            try:
                d = dict(runtime.iter_info[-1])
            except IndexError:
                d = dict(self.runtime)
            d['run'] = runtime.run
            out = os.path.abspath(os.path.expanduser(path % d))
        else:
            out = os.path.abspath(os.path.expanduser(path % self.runtime))

        return out

############
# TESTING ##
############

if __name__ == "__main__":
    pa = Paths()
    print(pa.auto_file())
    print(pa.plot_file())
    print(pa.recon_file())
