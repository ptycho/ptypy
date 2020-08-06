# -*- coding: utf-8 -*-
"""
Util sub-package

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
from .misc import *
from .math_utils import *
from .array_utils import *
from .scripts import *
from .parameters import Param, asParam
from .verbose import *
from .citations import *
from . import descriptor
from . import parallel
from .. import __has_matplotlib__ as hmpl
if hmpl:
    from .plot_utils import *
    from .plot_client import PlotClient, MPLClient, spawn_MPLClient, MPLplotter
del hmpl

SUBPIXEL_SHIFT_METHODS = {'bicubic': shift_interp, 'fourier': shift_fourier, 'linear': lambda x, y: shift_interp(x, y, order=1)}
