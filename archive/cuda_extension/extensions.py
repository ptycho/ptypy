'''
These are the optional extensions for ptypy
'''


from distutils.version import LooseVersion
from distutils.extension import Extension
import os
import multiprocessing
import subprocess
import re
import numpy as np


# this is a hacky version, but is the desired behaviour
class AccelerationExtension(object):
    def __init__(self, debug=False):
        self.debug = debug
        self._options = None

    def get_full_options(self):
        return self._options

    def get_reflection_options(self):
        user_options = []
        boolean_options = []
        for name, description in self._options.items():
            if isinstance(description['default'], str):
                user_options.append((name+'=', None, description['doc']))
            elif isinstance(description['default'], bool):
                user_options.append((name, None, description['doc']))
                boolean_options.append(name)
            else:
                raise NotImplementedError("Don't know what to do with parameter:%s of type: %s" % (name, type(description['default'])))
        return user_options, boolean_options

    def build(self, options):
        raise NotImplementedError('You need to implement the build method!')

    def getExtension(self):
        raise NotImplementedError('You need to return cython extension object.')


class CudaExtension(AccelerationExtension): # probably going to inherit from something.
    def __init__(self, *args, **kwargs):
        super(CudaExtension, self).__init__(*args, **kwargs)
        self._options = {'cudadir': {'default': '',
                                     'doc': 'CUDA directory'},
                         'cudaflags': {'default': '-gencode arch=compute_35,\\"code=sm_35\\" ' +
                                                  '-gencode arch=compute_37,\\"code=sm_37\\" ' +
                                                  '-gencode arch=compute_52,\\"code=sm_52\\" ' +
                                                  '-gencode arch=compute_60,\\"code=sm_60\\" ' +
                                                  '-gencode arch=compute_70,\\"code=sm_70\\" ',
                                       'doc': 'Flags to the CUDA compiler'},
                         'gputiming': {'default': False,
                                       'doc': 'Do GPU timing'}}

    def build(self, options):
        cudadir = options['cudadir']
        cudaflags = options['cudaflags']
        gputiming = options['gputiming']
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
            '-DCMAKE_CUDA_FLAGS={}'.format(cudaflags),
            '-DGPU_TIMING={}'.format("ON" if gputiming else "OFF")
        ]
        if cudadir:
            cmake_args += '-DCMAKE_CUDA_COMPILER="{}/bin/nvcc"'.format(cudadir)
        build_args = ["--config", "Debug" if self.debug else "Release", "--", "-j{}".format(multiprocessing.cpu_count() + 1)]
        if not os.path.exists(buildtmp):
            os.makedirs(buildtmp)
        env = os.environ.copy()
        subprocess.check_call(['cmake', srcdir] + cmake_args,
                              cwd=buildtmp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=buildtmp)
        print("Complete.")

    def getExtension(self):
        libdirs = ['build/cuda']
        if 'LD_LIBRARY_PATH' in os.environ:
            libdirs += os.environ['LD_LIBRARY_PATH'].split(':')
        return Extension('*',
                         sources=['ptypy/accelerate/cuda/gpu_extension.pyx'],
                         include_dirs=[np.get_include()],
                         libraries=['gpu_extension', 'cudart', 'cufft'],
                         library_dirs=libdirs,
                         depends=['build/cuda/libgpu_extension.a', ],
                         language="c++")
