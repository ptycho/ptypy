#!/usr/bin/env python

import setuptools

# exclude_packages = ["test.*", "test"]
# package_list = setuptools.find_packages(exclude=exclude_packages)
# print(package_list)
setuptools.setup(
    #name='Python Ptychography toolbox',
    #version=VERSION,
    #author='Pierre Thibault, Bjoern Enders, Martin Dierolf and others',
    #description='',
    #long_description=open('README.rst', 'r').read(),
    # package_dir={'ptypy': 'ptypy'},
    # packages=package_list,
    # package_data={'ptypy': ['resources/*',],
    #               'ptypy.accelerate.cuda_pycuda.cuda': ['*.cu']},
    scripts=['scripts/ptypy.plot',
             'scripts/ptypy.inspect',
             'scripts/ptypy.plotclient',
             'scripts/ptypy.new',
             'scripts/ptypy.csv2cp',
             'scripts/ptypy.run'],
)
