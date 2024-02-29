'''
Compilation tools for Nvidia builds of extension modules.
'''
import os, re
import subprocess
import sysconfig
from distutils.unixccompiler import UnixCCompiler
from distutils.command.build_ext import build_ext


def find_in_path(name, path):
    "Find a file in a search path"
    # adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = os.path.join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    """
    Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """
    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = os.path.join(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                                   'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': os.path.join(home, 'include'),
                  'lib64': os.path.join(home, 'lib64')}

    # If lib64 does not exist, try lib instead (as common in conda env)
    if not os.path.exists(cudaconfig['lib64']):
        cudaconfig['lib64'] = os.path.join(home, 'lib')
    
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))
    return cudaconfig

def get_cuda_version(nvcc):
    """
    Get the CUDA version py running `nvcc --version`.
    """
    stdout = subprocess.check_output([nvcc,"--version"]).decode("utf-8")
    if bool(stdout.rstrip()):
        regex = r'release (\S+),'
        match = re.search(regex, stdout)
        if match:
            return float(match.group(1))
        raise LookupError('Unable to parse nvcc version output from {}'.format(stdout))
    else:
        return None 

def get_cuda_arch_flags(version):
    if version in (10.0, 10.1, 10.2):
        archflag = ' -gencode=arch=compute_60,code=sm_60' + \
                   ' -gencode=arch=compute_61,code=sm_61' + \
                   ' -gencode=arch=compute_70,code=sm_70' + \
                   ' -gencode=arch=compute_75,code=sm_75' + \
                   ' -gencode=arch=compute_75,code=compute_75'
    elif version == 11.0:
        archflag = ' -gencode=arch=compute_60,code=sm_60' + \
                   ' -gencode=arch=compute_61,code=sm_61' + \
                   ' -gencode=arch=compute_70,code=sm_70' + \
                   ' -gencode=arch=compute_75,code=sm_75' + \
                   ' -gencode=arch=compute_80,code=sm_80' + \
                   ' -gencode=arch=compute_80,code=compute_80'
    elif version >= 11.1:
        archflag = ' -gencode=arch=compute_60,code=sm_60' + \
                   ' -gencode=arch=compute_61,code=sm_61' + \
                   ' -gencode=arch=compute_70,code=sm_70' + \
                   ' -gencode=arch=compute_75,code=sm_75' + \
                   ' -gencode=arch=compute_80,code=sm_80' + \
                   ' -gencode=arch=compute_86,code=sm_86' + \
                   ' -gencode=arch=compute_86,code=compute_86'
    else:
        raise ValueError("CUDA version %s not supported" %str(version))
    return archflag

class NvccCompiler(UnixCCompiler):
    def __init__(self, *args, **kwargs):
        super(NvccCompiler, self).__init__(*args, **kwargs)
        self.CUDA = locate_cuda()
        self.CUDA_VERSION = get_cuda_version(self.CUDA["nvcc"])
        module_dir = os.path.join(__file__.strip('import_fft.py'), 'cuda', 'filtered_fft') 
        # by default, compile for all of these
        archflag = get_cuda_arch_flags(self.CUDA_VERSION)
        self.src_extensions.append('.cu')
        self.LD_FLAGS = [archflag, "-lcufft_static", "-lculibos", "-ldl", "-lrt", "-lpthread", "-cudart shared"]
        self.NVCC_FLAGS = ["-dc", archflag]
        self.CXXFLAGS = ['"-fPIC"']
        import pybind11
        pybind_includes = [pybind11.get_include(), sysconfig.get_path('include')]  
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

class CustomBuildExt(build_ext):

    def build_extension(self, ext):
        has_cu = any([src.endswith('.cu') for src in ext.sources])
        if has_cu:
            old_compiler = self.compiler
            self.compiler = NvccCompiler(verbose=old_compiler.verbose,
                                        dry_run=old_compiler.dry_run,
                                        force=old_compiler.force) # this is our bespoke compiler
            super(CustomBuildExt, self).build_extension(ext)
            self.compiler=old_compiler
        else:
            super(CustomBuildExt, self).build_extension(ext)


