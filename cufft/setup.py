#!/usr/bin/env python

# we should aim to remove the distutils dependency
import setuptools
from distutils.core import setup, Extension
import os

ext_modules = []
cmdclass = {}
# filtered Cuda FFT extension module
try:
    from extensions import locate_cuda, get_cuda_version # this raises an error if pybind11 is not available
    CUDA = locate_cuda() # this raises an error if CUDA is not available
    CUDA_VERSION = get_cuda_version(CUDA['nvcc'])
    if CUDA_VERSION < 10:
        raise ValueError("filtered cufft requires CUDA >= 10")
    from extensions import CustomBuildExt
    cufft_dir = "filtered_fft"
    ext_modules.append(
        Extension("filtered_cufft",
            sources=[os.path.join(cufft_dir, "module.cpp"),
                    os.path.join(cufft_dir, "filtered_fft.cu")]
        )
    )
    cmdclass = {"build_ext": CustomBuildExt}
    EXTBUILD_MESSAGE = "The filtered cufft extension has been successfully installed.\n"
except EnvironmentError as e:
    EXTBUILD_MESSAGE = '*' * 75 + "\n"
    EXTBUILD_MESSAGE += "Could not install the filtered cufft extension.\n"
    EXTBUILD_MESSAGE += "Make sure to have CUDA >= 10 installed.\n"
    EXTBUILD_MESSAGE += '*' * 75 + "\n"
    EXTBUILD_MESSAGE += 'Error message: ' + str(e)
except ImportError as e:
    EXTBUILD_MESSAGE = '*' * 75 + "\n"
    EXTBUILD_MESSAGE += "Could not install the filtered cufft extension.\n"
    EXTBUILD_MESSAGE += "Make sure to have pybind11 installed.\n"
    EXTBUILD_MESSAGE += '*' * 75 + "\n"
    EXTBUILD_MESSAGE += 'Error message: ' + str(e)
    
exclude_packages = []
package_list = setuptools.find_packages(exclude=exclude_packages)
setup(
    name='filtered cufft',
    version=0.1,
    author='Bjoern Enders, Benedikt Daurer, Joerg Lotze',
    description='Extension of CuFFT to include pre- and post-filters using callbacks',
    packages=package_list,
    ext_modules=ext_modules,
    install_requires=["pybind11"],
    cmdclass=cmdclass
)

print(EXTBUILD_MESSAGE)
