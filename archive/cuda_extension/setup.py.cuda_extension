#!/usr/bin/env python

import setuptools, setuptools.command.build_ext
from distutils.core import setup
#from Cython.Build import cythonize
import sys

from extensions import CudaExtension

CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
License :: OSI Approved
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Software Development
Operating System :: Unix
"""


MAJOR               = 0
MINOR               = 4
MICRO               = 1
ISRELEASED          = False
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


# import os
# if os.path.exists('MANIFEST'): os.remove('MANIFEST')

DEBUG = False

def write_version_py(filename='ptypy/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM ptypy/setup.py
short_version='%(version)s'
version='%(version)s'
release=%(isrelease)s

if not release:
    version += '.dev'
    import subprocess
    try:
        git_commit = subprocess.Popen(["git","log","-1","--pretty=oneline","--abbrev-commit"],
                     stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL).communicate()[0].split()[0]
    except:
        pass
    else:
        version += git_commit.strip().decode()

"""
    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION, 'isrelease': str(ISRELEASED)})
    finally:
        a.close()


if __name__ == '__main__':
    write_version_py()
    write_version_py('doc/version.py')
    try:
        execfile('ptypy/version.py')
        vers = version
    except:
        vers = VERSION


# optional packages that we don't always want to build
exclude_packages = ['*test*',
                    '*.accelerate.cuda*']

acceleration_build_steps = []

# I don't like this particularly, but I can't currently find a better way to give the desired result...
if '--tests' in sys.argv:
    sys.argv.remove('--tests')
    exclude_packages.remove('*test*')

if '--with-cuda' in sys.argv:
    sys.argv.remove('--with-cuda')
    acceleration_build_steps.append(CudaExtension(DEBUG))
    exclude_packages.remove('*.accelerate.cuda*')

if '--all-acceleration' in sys.argv:
    sys.argv.remove('--all-acceleration')
    # cuda
    acceleration_build_steps.append(CudaExtension(DEBUG))
    exclude_packages.remove('*.accelerate.cuda*')
    #exclude_packages.remove('*array_based*')


# chain this before build_ext
class BuildExtAcceleration(setuptools.command.build_ext.build_ext):
    """Custom build command, extending the build with CUDA / Cmake."""
    # add the build parameters via reflection for each extension.
    for ext in acceleration_build_steps:
        user_options, boolean_options = ext.get_reflection_options()
        setuptools.command.build_ext.build_ext.user_options.append(user_options)
        setuptools.command.build_ext.build_ext.boolean_options.append(boolean_options)

    def initialize_options(self):
        # initialise the options for each extension
        setuptools.command.build_ext.build_ext.initialize_options(self)
        for ext in acceleration_build_steps:
            for key, desc in ext.get_full_options().items():
                self.__dict__[key] = desc['default']

    def run(self):
        # run the build for each extension
        for ext in acceleration_build_steps:
            options = {}
            for key, desc in ext.get_full_options().items():
                options[key] = self.__dict__[key]
            ext.build(options)
        setuptools.command.build_ext.build_ext.run(self)


extensions = [ext.getExtension() for ext in acceleration_build_steps]

package_list = setuptools.find_packages(exclude=exclude_packages)
#print(package_list)
setup(
    name='Python Ptychography toolbox',
    version=VERSION,
    author='Pierre Thibault, Bjoern Enders, Martin Dierolf and others',
    description='Ptychographic reconstruction toolbox',
    long_description=open('README.rst', 'r').read(),
    package_dir={'ptypy': 'ptypy'},
    packages=package_list,
    package_data={'ptypy': ['resources/*',],
                  'ptypy.accelerate.py_cuda.cuda': ['*.cu'],
                  'ptypy.accelerate.py_cuda.cuda.filtered_fft': ['*.hpp', '*.cpp', 'Makefile', '*.cu', '*.h']},
    scripts=['scripts/ptypy.plot',
             'scripts/ptypy.inspect',
             'scripts/ptypy.plotclient',
             'scripts/ptypy.new',
             'scripts/ptypy.csv2cp',
             'scripts/ptypy.run'],
    #ext_modules=cythonize(extensions),
    #cmdclass={'build_ext': BuildExtAcceleration
    #}
)
