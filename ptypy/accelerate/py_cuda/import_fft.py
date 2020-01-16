# monkey-patch setuptools
from distutils import sysconfig
import os
import shutil
from pycuda import driver

def replace_flags(flags):
    ret = []
    bflag=False
    for f in flags:
        if bflag:
            ret += ['-Xcompiler', '"-B ' + f + '"']
            bflag = False
        elif f.startswith('-Wl'):
            ret += ['-Xlinker', f.replace('-Wl,', '')]
        elif f == '-Wstrict-prototypes':  # C only
            continue
        elif f.startswith('-W'):
            ret += ['-Xcompiler', f]
        elif f.startswith('-f'):
            ret += ['-Xcompiler', f]
        elif f == '-pthread':
            ret.append('-lpthread')
        elif f == '-B':
            bflag=True
        else:
            ret.append(f)
    return ret
        
def get_customize_compiler(rows, columns, old):
    
    cmp = driver.Context.get_device().compute_capability()
    #print(dev.compute_capability())
    archflag = '-arch=sm_{}{}'.format(cmp[0], cmp[1])

    def customize_compiler(compiler):
        old(compiler)
        #print(compiler.compiler)
        #print(compiler.compiler_so)
        #print(compiler.compiler_cxx)
        #print(compiler.linker_so)

        comp_cmd = replace_flags(compiler.compiler) + ['-dc', '-x', 'cu', archflag]
        comp_so = replace_flags(compiler.compiler_so) + ['-Xcompiler', '-fPIC', '-dc', '-x', 'cu', archflag]
        comp_cxx = replace_flags(compiler.compiler_cxx)
        linker_so = replace_flags(compiler.linker_so) + [archflag]
        
        mod = 'module_' + str(rows) + '_' + str(columns)
        defines = [
            '-DMODULE_NAME=' + mod,
            '-DMY_FFT_ROWS=' + str(rows),
            '-DMY_FFT_COLS=' + str(columns)
        ]

        comp_cxx += defines
        comp_cmd += defines
        comp_so += defines

        comp_cmd[0] = 'nvcc'
        comp_so[0] = 'nvcc'
        comp_cxx[0] = 'nvcc'
        linker_so[0] = 'nvcc'


        compiler.set_executables(
            compiler=comp_cmd,
            compiler_so = comp_so,
            compiler_cxx = comp_cxx,
            linker_so=linker_so)

        #print(compiler.compiler)
        #print(compiler.compiler_so)
        #print(compiler.compiler_cxx)
        #print(compiler.linker_so)
    
    return customize_compiler
    
def import_fft(rows, columns):

    module_name = 'module_' + str(rows) + '_' + str(columns)
    dirname = os.path.join(os.path.dirname(__file__), 'cuda', 'filtered_fft')
    src = os.path.join(dirname, 'module.cpp')
    dst = os.path.join(dirname, module_name + '.cpp')
    shutil.copy(src, dst)
    #print('copies {} to {}'.format(src, dst))

    # monkey-patch the customize_compiler function
    old = sysconfig.customize_compiler
    sysconfig.customize_compiler = get_customize_compiler(rows, columns, old)

    import cppimport
    cppimport.set_quiet(True)
    cppimport
    filtered_fft = cppimport.imp("ptypy.accelerate.py_cuda.cuda.filtered_fft." + module_name)
    
    # revert the monkey-patch
    sysconfig.customize_compiler = old

    return filtered_fft

