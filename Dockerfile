# Select MPI environment: openmpi or mpich
ARG MPI=openmpi

# Select Platform: core, full, pycuda or cupy
ARG PLATFORM=cupy

# Select CUDA version
ARG CUDAVERSION=12.4

# Pull from mambaforge and install XML and ssh
FROM condaforge/mambaforge AS base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y libxml2 ssh

# Pull from base image and install OpenMPI/MPICH
FROM base AS mpi
ARG MPI
RUN mamba install -n base -c conda-forge ${MPI}

# Pull from MPI build install core dependencies
FROM base AS core
COPY ./dependencies_core.yml ./dependencies.yml
RUN mamba env update -n base -f dependencies.yml

# Pull from MPI build and install full dependencies
FROM mpi AS full
COPY ./dependencies_full.yml ./dependencies.yml
RUN mamba env update -n base -f dependencies.yml

# Pull from MPI build and install accelerate/pycuda dependencies
FROM mpi AS pycuda
ARG CUDAVERSION
COPY ./ptypy/accelerate/cuda_pycuda/dependencies.yml ./dependencies.yml
COPY ./cufft/dependencies.yml ./dependencies_cufft.yml
RUN mamba install cuda-version=${CUDAVERSION} && \
    mamba env update -n base -f dependencies.yml && \
    mamba env update -n base -f dependencies_cufft.yml

# Pull from MPI build and install accelerate/cupy dependencies
FROM mpi AS cupy
ARG CUDAVERSION
COPY ./ptypy/accelerate/cuda_cupy/dependencies.yml ./dependencies.yml
COPY ./cufft/dependencies.yml ./dependencies_cufft.yml
RUN mamba install cuda-version=${CUDAVERSION} && \
    mamba env update -n base -f dependencies.yml && \
    mamba env update -n base -f dependencies_cufft.yml

# Pull from platform specific image and install ptypy 
FROM ${PLATFORM} AS build
COPY . .
RUN mamba run pip install .

# For core/full build, no post processing needed
FROM build AS core-post
FROM build AS full-post

# For pycuda build, install filtered cufft
FROM build AS pycuda-post
RUN mamba run pip install ./cufft

# For pycuda build, install filtered cufft
FROM build AS cupy-post
RUN mamba run pip install ./cufft

# Platform specific runtime container
FROM ${PLATFORM}-post AS runtime

# Run PtyPy run script as entrypoint
ENTRYPOINT ["ptypy.cli"]
