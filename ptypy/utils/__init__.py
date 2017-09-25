# -*- coding: utf-8 -*-
"""
Util sub-package

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
from misc import *
from math_utils import *
from array_utils import *
from scripts import *
from parameters import Param, asParam
from verbose import *
#import propagation as prop
#from embedded_shell import ipshell
#import ipython_kernel

from .. import __has_matplotlib__ as hmpl
# from .. import __has_PyQt4__ as hpqt4


if hmpl:# and hpqt4:
    from plot_utils import *
    from plot_client import PlotClient, MPLClient, spawn_MPLClient, MPLplotter
del hmpl
# del hpqt4
import validator

import parallel

