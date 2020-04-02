# # monkey-patch setuptools
# from distutils import sysconfig
# import os
# import shutil
import os
from distutils.sysconfig import get_config_var
from distutils.unixccompiler import UnixCCompiler
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

class NvccCompiler(UnixCCompiler):
    def __init__(self, *args, **kwargs):
        super(NvccCompiler, self).__init__(*args, **kwargs)
        self.CUDA = locate_cuda()
        module_dir = os.path.join(__file__.strip('import_fft.py'), 'cuda', 'filtered_fft') 
        cmp = driver.Context.get_device().compute_capability()
        archflag = '-arch=sm_{}{}'.format(cmp[0], cmp[1])
        self.src_extensions.append('.cu')
        self.LD_FLAGS = ["-lcufft_static", "-lculibos", "-ldl", "-lrt", "-lpthread", "-cudart shared"]
        self.NVCC_FLAGS = ["-dc", archflag]
        self.CXXFLAGS = ['"-fPIC"']
        pybind_includes = [pybind11.get_include(), pybind11.get_include(True)+'/python3.7m/']  # why has pybind includes stopped giving me the python3,7m directory?
        INCLUDES = pybind_includes + [self.CUDA['lib64'], module_dir]
        self.INCLUDES = ["-I%s" % ix for ix in INCLUDES]
        self.OPTFLAGS = ["-O3", "-std=c++14"]

    def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        default_compiler_so = self.compiler_so
        CPPFLAGS = self.INCLUDES + extra_postargs # little hack here, since postargs usually goes at the end, which we won't do.
        # makefile line is
        # $(NVCC) $(NVCC_FLAGS) $(OPTFLAGS) -Xcompiler "$(CXXFLAGS)" $(CPPFLAGS)
        compiler_command = [self.CUDA["nvcc"]] + self.NVCC_FLAGS + self.OPTFLAGS + ["-Xcompiler"] + self.CXXFLAGS + CPPFLAGS
        compiler_exec = " ".join(compiler_command)
        self.set_executable('compiler_so', compiler_exec)
        postargs = [] # we don't actually have any postargs
        super(NvccCompiler, self)._compile(obj, src, ext, cc_args, postargs, pp_opts) # the _compile method
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so
    
    def link(self, target_desc, objects,
             output_filename, output_dir=None, libraries=None,
             library_dirs=None, runtime_library_dirs=None,
             export_symbols=None, debug=0, extra_preargs=None,
             extra_postargs=None, build_temp=None, target_lang=None):
        default_linker_so = self.linker_so
        # make file line is
        # $(NVCC) $(OPTFLAGS) -shared $(LD_FLAGS) $(OBJ) $(OBJ_MOD) -o $@
        linker_command = [self.CUDA["nvcc"]] + self.OPTFLAGS + ["-shared"] + self.LD_FLAGS
        linker_exec = " ".join(linker_command)
        self.set_executable('linker_so', linker_exec)
        super(NvccCompiler, self).link(target_desc, objects,
             output_filename, output_dir=None, libraries=None,
             library_dirs=None, runtime_library_dirs=None,
             export_symbols=None, debug=0, extra_preargs=None,
             extra_postargs=None, build_temp=None, target_lang=None)
        self.linker_so = default_linker_so


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        old_compiler = self.compiler
        self.compiler = NvccCompiler(verbose=old_compiler.verbose,
                                    dry_run=old_compiler.dry_run,
                                    force=old_compiler.force) # this is our bespoke compiler
        build_ext.build_extensions(self)
        self.compiler=old_compiler


def import_fft(rows, columns, build_path=None):
    if build_path is None:
        build_path = tempfile.mkdtemp(prefix="extension_tests")

    full_module_name = "module"
    module_dir = os.path.join(__file__.strip('import_fft.py'), 'cuda', 'filtered_fft')  
    # do I need to add all the arguments here, or do they default to empty lists/None?
    ext = Extension(full_module_name,
                    sources=[os.path.join(module_dir, "module.cpp"),
                             os.path.join(module_dir, "filtered_fft.cu")],
                    library_dirs=[],
                    libraries=[], # distuils adds a -l infront of all of these (add_library_option:https://github.com/python/cpython/blob/1c1e68cf3e3a2a19a0edca9a105273e11ddddc6e/Lib/distutils/ccompiler.py#L1115)
                    runtime_library_dirs=[],
                    extra_compile_args=["-DMY_FFT_COLS=%s" % str(columns) , "-DMY_FFT_ROWS=%s" % str(rows)],
                    include_dirs=[])

    script_args = ['build_ext',
                   '--build-temp=%s' % build_path,
                   '--build-lib=%s' % build_path]
    # do I need full_module_name here?
    setuptools_args = {"name": full_module_name,
                       "ext_modules": [ext],
                       "script_args": script_args,
                       "cmdclass":{"build_ext": custom_build_ext}
                       }

    setuptools.setup(**setuptools_args)

    spec = importlib.util.spec_from_file_location(full_module_name,
                                                  os.path.join(build_path,
                                                               "module" + get_config_var('EXT_SUFFIX')
                                                               )
                                                  )

    return importlib.util.module_from_spec(spec)
