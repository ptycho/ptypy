# -*- coding: utf-8 -*-
"""
Debug sub-package.

A separate module because all this IPython mingling repeatedly caused
trouble. Should not have any dependence on other submodules

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

from .embedded_shell import ipshell
from . import ipython_kernel


