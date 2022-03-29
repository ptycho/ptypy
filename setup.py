#!/usr/bin/env python

# we should aim to remove the distutils dependency
import setuptools
from distutils.core import setup
import sys

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
MINOR               = 5
MICRO               = 0
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
        exec(open('ptypy/version.py').read())
        vers = version # noqa: F821
    except:
        vers = VERSION

exclude_packages = ["test.*", "test"]
package_list = setuptools.find_packages(exclude=exclude_packages)
setup(
    name='Python Ptychography toolbox',
    version=VERSION,
    author='Pierre Thibault, Bjoern Enders, Martin Dierolf and others',
    description='Ptychographic reconstruction toolbox',
    long_description=open('README.rst', 'r').read(),
    package_dir={'ptypy': 'ptypy'},
    packages=package_list,
    package_data={'ptypy': ['resources/*',],
                  'ptypy.accelerate.cuda_pycuda.cuda': ['*.cu']},
    scripts=['scripts/ptypy.plot',
             'scripts/ptypy.inspect',
             'scripts/ptypy.plotclient',
             'scripts/ptypy.new',
             'scripts/ptypy.csv2cp',
             'scripts/ptypy.run'],
)
