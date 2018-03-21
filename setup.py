#!/usr/bin/env python

import distutils
import setuptools
from distutils.core import setup, Extension
from distutils.version import LooseVersion
from Cython.Build import cythonize
import numpy as np
import re
import os
import multiprocessing

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
MINOR               = 2
MICRO               = 0
ISRELEASED          = False
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

#import os
#if os.path.exists('MANIFEST'): os.remove('MANIFEST')


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
        version += git_commit.strip()

"""
    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION, 'isrelease': str(ISRELEASED)})
    finally:
        a.close()

if __name__ == '__main__':
    write_version_py()
    try:
        execfile('ptypy/version.py')
        vers = version
    except:
        vers = VERSION

libdirs = ['build/cuda']
if 'LD_LIBRARY_PATH' in os.environ:
    libdirs += os.environ['LD_LIBRARY_PATH'].split(':')

extensions = [
    Extension(            
        '*',
        sources=['ptypy/gpu/gpu_extension.pyx'],
        include_dirs=[np.get_include()],
        libraries=[
            'gpu_extension', 
            'cudart', 'cufft'],
        library_dirs=libdirs,
        depends=[
            'build/cuda/libgpu_extension.a',
            #'cuda/*.cu',
            #'cuda/*.h',
            #'cuda/CMakeLists.txt'
            ],
        language="c++"
    )
]

# chain this before build_ext
class BuildExtCudaCommand(setuptools.command.build_ext.build_ext):
    """Custom build command, extending the build with CUDA / Cmake."""

    user_options = setuptools.command.build_ext.build_ext.user_options + \
        [
            ('cudadir=', None, 'CUDA directory'),
            ('cudaflags=', None, 'Flags to the CUDA compiler'),
            ('gputiming', None, 'Do GPU timing')
        ]
    boolean_options = setuptools.command.build_ext.build_ext.boolean_options + \
        ['gputiming']

    def initialize_options(self):
        setuptools.command.build_ext.build_ext.initialize_options(self)
        self.cudadir = ''
        self.cudaflags = '-gencode arch=compute_35,\\"code=sm_35\\" ' + \
                         '-gencode arch=compute_37,\\"code=sm_37\\" ' + \
                         '-gencode arch=compute_60,\\"code=sm_60\\" ' + \
                         '-gencode arch=compute_70,\\"code=sm_70\\" ' + \
                         '-gencode arch=compute_70,\\"code=compute_70\\"'
        self.gputiming = False

    def run(self):
        #print "----------{}-------".format(self.build_temp)
        self.run_cuda_cmake()
        setuptools.command.build_ext.build_ext.run(self)


    def run_cuda_cmake(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the CUDA extensions.")

        cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                     out.decode()).group(1))
        if cmake_version < '3.8.0':
            raise RuntimeError("CMake >= 3.8.0 is required")

        srcdir = os.path.abspath('cuda')
        buildtmp = os.path.abspath(os.path.join('build', 'cuda'))
        cmake_args = [
            "-DCMAKE_BUILD_TYPE=" + ("Debug" if self.debug else "Release"),
            '-DCMAKE_CUDA_FLAGS={}'.format(self.cudaflags),
            '-DGPU_TIMING={}'.format("ON" if self.gputiming  else "OFF")
        ]
        if self.cudadir:
            cmake_args += '-DCMAKE_CUDA_COMPILER="{}/bin/nvcc"'.format(self.cudadir)
        build_args = ["--config", "Debug" if self.debug else "Release", "--", "-j{}".format(multiprocessing.cpu_count() + 1)]
        if not os.path.exists(buildtmp):
            os.makedirs(buildtmp)
        env = os.environ.copy()
        subprocess.check_call(['cmake', srcdir] + cmake_args,
                              cwd=buildtmp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=buildtmp)
        print ""

setup(
    name='Python Ptychography toolbox',
    version=VERSION,
    author='Pierre Thibault, Bjoern Enders, Martin Dierolf and others',
    description='Ptychographic reconstruction toolbox',
    long_description=file('README.rst', 'r').read(),
    #install_requires = ['numpy>=1.8',\
                        #'h5py>=2.2',\
                        #'matplotlib>=1.3',\
                        #'pyzmq>=14.0',\
                        #'scipy>=0.13',\
                        #'mpi4py>=1.3'],
    package_dir={'ptypy': 'ptypy'},
    packages=['ptypy',
              'ptypy.core',
              'ptypy.debug',
              'ptypy.utils',
              'ptypy.simulations',
              'ptypy.engines',
              'ptypy.io',
              'ptypy.resources',
              'ptypy.experiment',
              'ptypy.test',
              'ptypy.gpu',
              'ptypy.array_based'],
    package_data={'ptypy': ['resources/*', ]},
    #include_package_data=True
    scripts=[
        'scripts/ptypy.plot',
        'scripts/ptypy.inspect',
        'scripts/ptypy.plotclient',
        'scripts/ptypy.new',
        'scripts/ptypy.csv2cp'
    ],
    ext_modules=cythonize(
        extensions        
    ),
    cmdclass = {
        'build_ext' : BuildExtCudaCommand
    }
)
