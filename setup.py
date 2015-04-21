#!/usr/bin/env python

from distutils.core import setup, Extension

setup(
    name = 'Python Ptychography toolbox',
    version = '0.1',
    author = 'Pierre Thibault, Bjoern Enders, Martin Dierolf and others',
    description = 'Ptychographic reconstruction toolbox', 
    long_description = file('README.md','r').read(),
    #install_requires = ['numpy>=1.8',\
                        #'h5py>=2.2',\
                        #'matplotlib>=1.3',\
                        #'pyzmq>=14.0',\
                        #'scipy>=0.13',\
                        #'mpi4py>=1.3'],
    package_dir = {'ptypy':'ptypy'},
    packages = ['ptypy',
                'ptypy.core',\
                'ptypy.utils',\
                'ptypy.simulations',\
                'ptypy.engines',\
                'ptypy.io',\
                'ptypy.resources',\
                'ptypy.experiment'],
    package_data={'ptypy':['resources/*',]},
    #include_package_data=True 
    )
