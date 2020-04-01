# # monkey-patch setuptools
# from distutils import sysconfig
# import os
# import shutil
import os
from distutils.sysconfig import get_config_var

import importlib
import setuptools
import setuptools.command.build_ext
import tempfile
from pycuda import driver

# from ptypy.utils.verbose import log
# on Windows, we need the original PATH without Anaconda's compiler in it:
PATH = os.environ.get('PATH')
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from . import find
import pybind11

import os
from os.path import join as pjoin
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
import numpy


def find_in_path(name, path):
    "Find a file in a search path"
    # adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                                   'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    default_linker_so = self.linker_so

    original__compile = self._compile
    original_link = self.link
    CUDA = locate_cuda()
    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        # if os.path.splitext(src)[1] == '.cpp':
            # use the cuda for .cu files
        full_module_name = "ptypy.accelerate.py_cuda.cuda.filtered_fft.module"
        module_file_path = find.find_module_cpppath(full_module_name)
    
        nvcc_path = find_in_path('nvcc', os.environ['PATH'])
        if nvcc_path is None or not os.path.isfile(nvcc_path):
            raise EnvironmentError('The nvcc binary could not be located in your'
                                    ' $PATH. Either add it to your path, or set'
                                    ' appropiate $CUDAHOME.')

        CUDADIR = os.path.dirname(nvcc_path)        
        module_dir = os.path.dirname(module_file_path)
        cmp = driver.Context.get_device().compute_capability()
        # print(dev.compute_capability())
        archflag = '-arch=sm_{}{}'.format(cmp[0], cmp[1])
        pybind_includes = [pybind11.get_include(True)]
        INCLUDES = pybind_includes + [CUDA['lib64'], module_dir]
        INCLUDES = ["-I%s" % ix for ix in INCLUDES]
        # PYMODEXT = $(shell python-config --extension-suffix)
        PYMODEXT = '.so'
        CPPFLAGS = INCLUDES + extra_postargs
        OPTFLAGS = ["-O3", "-std=c++14"]
        CXXFLAGS = ['"-fPIC"']
        NVCC_FLAGS = ["-dc", archflag]
        LD_FLAGS = ["-lcufft_static", "-lculibos", "-ldl", "-lrt", "-lpthread", "-cudart shared"]
        # $(NVCC) $(NVCC_FLAGS) $(OPTFLAGS) -Xcompiler "$(CXXFLAGS)" $(CPPFLAGS)
        compiler_command = [CUDA["nvcc"]] + NVCC_FLAGS + OPTFLAGS + ["-Xcompiler"] + CXXFLAGS + CPPFLAGS
        compiler_exec = " ".join(compiler_command)

        self.set_executable('compiler_so', compiler_exec)


        # use only a subset of the extra_postargs, which are 1-1 translated
        # from the extra_compile_args in the Extension class
        postargs = []
        pp = pp_opts.append([])
        # else:
        #     postargs = extra_postargs['gcc']

        original__compile(obj, src, ext, cc_args, postargs, pp) # the _compile method
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile

    def link(target_desc, objects,
             output_filename, output_dir=None, libraries=None,
             library_dirs=None, runtime_library_dirs=None,
             export_symbols=None, debug=0, extra_preargs=None,
             extra_postargs=None, build_temp=None, target_lang=None):
        OPTFLAGS = ["-O3", "-std=c++14"]
        LD_FLAGS = ["-lcufft_static", "-lculibos", "-ldl", "-lrt", "-lpthread", "-cudart shared"]

        linker_command = [CUDA["nvcc"]] + OPTFLAGS + ["-shared"] + LD_FLAGS
        linker_exec = " ".join(linker_command)

        self.set_executable('linker_so', linker_exec)
        original_link(target_desc, objects,
             output_filename, output_dir=None, libraries=None,
             library_dirs=None, runtime_library_dirs=None,
             export_symbols=None, debug=0, extra_preargs=None,
             extra_postargs=None, build_temp=None, target_lang=None)
        self.linker_so = default_linker_so
    self.link = link

# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


def import_fft(rows, columns, build_path=None):
    if build_path is None:
        build_path = tempfile.mkdtemp(prefix="extension_tests")


 # we only need this if it's not always in a fixed path. I think we can probably assume this.

    CUDA = locate_cuda()
    full_module_name = "ptypy.accelerate.py_cuda.cuda.filtered_fft.module"

    module_file_path = find.find_module_cpppath(full_module_name)
    module_dir = os.path.dirname(module_file_path)

    ext = Extension('module',
                    sources=[module_file_path, os.path.join(module_dir, "filtered_fft.cu")],
                    library_dirs=[],
                    libraries=[], # distuils adds a -l infront of all of these (add_library_option:https://github.com/python/cpython/blob/1c1e68cf3e3a2a19a0edca9a105273e11ddddc6e/Lib/distutils/ccompiler.py#L1115)
                    runtime_library_dirs=[],
                    # this syntax is specific to this build system
                    # we're only going to use certain compiler args with nvcc and not with gcc
                    # the implementation of this trick is in customize_compiler() below
                    extra_compile_args=["-DMY_FFT_COLS=%s" % str(columns) , "-DMY_FFT_ROWS=%s" % str(rows)],
                    include_dirs=[])



    script_args = ['build_ext',
                   '--build-temp=%s' % build_path,
                   '--build-lib=%s' % build_path]
    setuptools_args = {"name": full_module_name,
                       "ext_modules": [ext],
                       "script_args": script_args,
                       "cmdclass":{"build_ext": custom_build_ext}
                       }
    setuptools.setup(**setuptools_args)


    spec = importlib.util.spec_from_file_location('module',
                                                  os.path.join(build_path,
                                                               "module" + get_config_var('EXT_SUFFIX')
                                                               )
                                                  )
    mod = importlib.util.module_from_spec(spec)
    return mod
