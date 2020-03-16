# monkey-patch setuptools
from distutils import sysconfig
import os
import shutil
from pycuda import driver
from ptypy.utils.verbose import log


class CustomizeCompilerFactory:
    def __init__(self, rows, columns, old_customize_compiler_function):
        self.rows = rows
        self.columns = columns
        self.old_customize_compiler = old_customize_compiler_function
        log(2, "called get_customize_compiler with rows:%s, columns: %s" % (rows, columns))
        log(2, "and immediately self. rows:%s, self.columns: %s" % (self.rows, self.columns))
        cmp = driver.Context.get_device().compute_capability()
        #print(dev.compute_capability())
        self.archflag = '-arch=sm_{}{}'.format(cmp[0], cmp[1])
        mod = 'module_' + str(self.rows) + '_' + str(self.columns)
        log(2,"mod:%s" % mod)
        log(2, "self.rows:%s, columns:%s" % (self.rows, self.columns))
        self.defines = [
            '-DMODULE_NAME=' + mod,
            '-DMY_FFT_ROWS=' + str(self.rows),
            '-DMY_FFT_COLS=' + str(self.columns)
        ]

    def replace_flags(self, flags):
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

    def customize_compiler(self, compiler):
        self.old_customize_compiler(compiler)

        log(2, "rows:%s, cols: %s " % (self.rows, self.columns))
        comp_cmd = self.replace_flags(compiler.compiler) + ['-dc', '-x', 'cu', self.archflag]
        comp_so = self.replace_flags(compiler.compiler_so) + ['-Xcompiler', '-fPIC', '-dc', '-x', 'cu', self.archflag]
        comp_cxx = self.replace_flags(compiler.compiler_cxx)
        linker_so = self.replace_flags(compiler.linker_so) + [self.archflag]

        comp_cxx += self.defines
        comp_cmd += self.defines
        comp_so += self.defines
        log(2, "comp_cxx:%s" % str(comp_cxx))
        log(2, "comp_cmd:%s" % str(comp_cmd))
        log(2, "comp_so:%s" % str(comp_so))

        comp_cmd[0] = 'nvcc'
        comp_so[0] = 'nvcc'
        comp_cxx[0] = 'nvcc'
        linker_so[0] = 'nvcc'


        compiler.set_executables(
            compiler=comp_cmd,
            compiler_so = comp_so,
            compiler_cxx = comp_cxx,
            linker_so=linker_so)


def import_fft(rows, columns):

    module_name = 'module_' + str(rows) + '_' + str(columns)
    dirname = os.path.join(os.path.dirname(__file__), 'cuda', 'filtered_fft')
    src = os.path.join(dirname, 'module.cpp')
    dst = os.path.join(dirname, module_name + '.cpp')
    shutil.copy(src, dst)
    #print('copies {} to {}'.format(src, dst))

    # monkey-patch the customize_compiler function
    old = sysconfig.customize_compiler
    new = CustomizeCompilerFactory(rows, columns, old)
    print("The new.rows:%s, new.columns: %s" % (new.rows, new.columns))
    log(2, "new:%s" % new)
    sysconfig.customize_compiler = new.customize_compiler

    import cppimport
    cppimport.set_quiet(True)
    cppimport
    filtered_fft = cppimport.imp("ptypy.accelerate.py_cuda.cuda.filtered_fft." + module_name)
    
    # revert the monkey-patch
    sysconfig.customize_compiler = old
    del new
    return filtered_fft

