# -*- coding: utf-8 -*-
"""
Build script for the compiled parts of ptypy. All package metadata lives in
pyproject.toml; this file only declares the C extensions.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import os
import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.errors import CCompilerError, ExecError, PlatformError

# Set PTYPY_REQUIRE_EXT=1 to turn a failed extension build into a failed
# install. Used by the CI, where a silent failure would go unnoticed.
REQUIRED = os.environ.get("PTYPY_REQUIRE_EXT", "") not in ("", "0")

BUILD_ERRORS = (CCompilerError, ExecError, PlatformError)

WARNING = """
***************************************************************************
WARNING: the %s C extension could not be built, so the
         corresponding ptypy features will not be available.
         Install a C compiler and reinstall ptypy to enable them.
***************************************************************************
"""


class optional_build_ext(build_ext):
    """Build the C extensions if we can, carry on with a warning if we can't."""

    def run(self):
        try:
            build_ext.run(self)
        except BUILD_ERRORS:
            if REQUIRED:
                raise
            self._warn("")

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except BUILD_ERRORS:
            if REQUIRED:
                raise
            self._warn(ext.name)

    def _warn(self, name):
        sys.stderr.write(WARNING % name)
        sys.stderr.flush()


unwrap_dir = os.path.join("ptypy", "utils", "unwrap")

ext_modules = [
    Extension("ptypy.utils.unwrap._qmunwrap",
              sources=[os.path.join(unwrap_dir, "_qmunwrapmodule.c"),
                       os.path.join(unwrap_dir, "_qmunwrap.c")],
              libraries=[] if os.name == "nt" else ["m"]),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": optional_build_ext})
