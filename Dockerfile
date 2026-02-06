# Select MPI environment: openmpi or mpich
ARG MPI=openmpi

# Select Platform: core, full, pycuda or cupy
ARG PLATFORM=cupy

# Select CUDA version
ARG CUDAVERSION=12.8

# Pull from mambaforge and install XML and ssh
FROM condaforge/mambaforge AS base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y libxml2 ssh

# Pull from base image and install OpenMPI/MPICH
FROM base AS mpi
ARG MPI
RUN mamba install -n base -c conda-forge ${MPI}

# Pull from MPI build and install core dependencies
FROM base AS core
RUN mamba install -n base -y -c conda-forge \
    python numpy scipy h5py pip

# Pull from MPI build and install full dependencies
FROM mpi AS full
ARG MPI
RUN mamba install -n base -y -c conda-forge \
    python numpy scipy matplotlib h5py \ 
    pyzmq mpi4py[build=*${MPI}*] packaging \
    pillow pyfftw pyyaml pip

# Pull from MPI build and install accelerate/pycuda dependencies
FROM mpi AS pycuda
ARG CUDAVERSION MPI
RUN mamba install -n base -y -c conda-forge -c nvidia \
    python numpy scipy matplotlib h5py pyzmq mpi4py[build=*${MPI}*] \
    pillow pyfftw pyyaml compilers pip \
    reikna pycuda cuda-nvcc cuda-cudart-dev cuda-version=${CUDAVERSION}

# Pull from MPI build and install accelerate/cupy dependencies
FROM mpi AS cupy
ARG CUDAVERSION MPI
RUN mamba install -n base -y -c conda-forge \
    python numpy scipy matplotlib h5py pyzmq mpi4py[build=*${MPI}*] \
    pillow pyfftw pyyaml compilers pip \
    cupy cuda-version=${CUDAVERSION}
RUN mamba clean -y -a

# Pull from platform specific image and install ptypy 
FROM ${PLATFORM} AS build
COPY pyproject.toml ./
COPY ./templates ./templates
COPY ./benchmark ./benchmark
COPY ./ptypy ./ptypy
RUN pip install .

# For core build, clean up conda env
FROM build AS core-post
RUN mamba clean -y -a

# For full build, clean up conda env
FROM build AS full-post
RUN mamba clean -y -a

# For pycuda build, install filtered cufft
FROM build AS pycuda-post
ARG CUDAVERSION
RUN mamba install -n base -y -c conda-forge -c nvidia \
    python cmake>=3.8.0 pybind11 compilers \
    cuda-nvcc cuda-cudart-dev libcufft-dev libcufft-static cuda-version=${CUDAVERSION}
COPY ./cufft ./cufft
RUN pip install ./cufft
RUN mamba remove -n base -y \
    cmake pybind11 cuda-nvcc cuda-cudart-dev libcufft-dev libcufft-static
RUN mamba clean -y -a

# For cupy build, install filtered cufft
FROM build AS cupy-post
ARG CUDAVERSION
RUN mamba install -n base -y -c conda-forge -c nvidia \
    python cmake>=3.8.0 pybind11 compilers \
    cuda-nvcc cuda-cudart-dev libcufft-dev libcufft-static cuda-version=${CUDAVERSION}
COPY ./cufft ./cufft
RUN pip install ./cufft
RUN mamba remove -n base -y \
    cmake pybind11 libcufft-dev libcufft-static
RUN mamba clean -y -a

# Platform specific runtime container
FROM ${PLATFORM}-post AS runtime
RUN useradd --user-group ptypy-user
USER ptypy-user

# Run PtyPy run script as entrypoint
ENTRYPOINT ["ptypy.cli"]
