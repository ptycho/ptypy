#!/usr/bin/env python

from distutils.core import setup, Extension

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
        git_commit = subprocess.Popen(["git","log","-1","--pretty=oneline","--abbrev-commit"],stdout=subprocess.PIPE).communicate()[0].split()[0]
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

if __name__=='__main__':
    write_version_py()
    try:
        execfile('ptypy/version.py')
        vers= version
    except:
        vers = VERSION

setup(
    name = 'Python Ptychography toolbox',
    version = VERSION,
    author = 'Pierre Thibault, Bjoern Enders, Martin Dierolf and others',
    description = 'Ptychographic reconstruction toolbox', 
    long_description = file('README.rst','r').read(),
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
    scripts = [
        'scripts/ptypy.plot',
        'scripts/ptypy.inspect',
        'scripts/ptypy.plotclient',
        'scripts/ptypy.new'    
    ],
    )
