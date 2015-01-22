#!/usr/bin/env python

from distutils.core import setup, Extension

setup(
    name = 'Python Ptychography toolbox',
    version = '0.1',
    author = 'Pierre Thibault, Bjoern Enders, Martin Dierolf and others',
    description = 'Ptychographic reconstruction toolbox', 
    long_description = file('README.md','r').read(),
    package_dir = {'ptypy':'ptypy'},
    packages = ['ptypy',
                'ptypy.core',\
                'ptypy.utils',\
                'ptypy.simulations',\
                'ptypy.engines',\
                'ptypy.io',\
                'ptypy.experiment'],
    package_data={'ptypy':['resources/*',]},
    include_package_data=True 
    )
