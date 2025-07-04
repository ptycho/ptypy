# Select MPI environment: openmpi or mpich
ARG MPI=openmpi

# Select Platform: core, full, pycuda or cupy
ARG PLATFORM=cupy

# Select CUDA version
ARG CUDAVERSION=12.4

# Pull from mambaforge and install XML and ssh
FROM mambaorg/micromamba:2.0.8-debian12-slim AS base
# ENV DEBIAN_FRONTEND=noninteractive
# RUN apt-get update && apt-get install -y libxml2 ssh

# Pull from base image and install OpenMPI/MPICH
FROM base AS mpi
ARG MPI
RUN micromamba install -n base -c conda-forge ${MPI}

# Pull from MPI build install core dependencies
FROM base AS core
COPY ./dependencies_core.yml ./dependencies.yml
RUN micromamba env update -n base -f dependencies.yml && \
    micromamba clean -y --all --force-pkgs-dirs

# Pull from MPI build and install full dependencies
FROM mpi AS full
COPY ./dependencies_full.yml ./dependencies.yml
RUN micromamba env update -n base -f dependencies.yml && \
    micromamba clean -y --all --force-pkgs-dirs

# Pull from MPI build and install accelerate/pycuda dependencies
FROM mpi AS pycuda
ARG CUDAVERSION
COPY ./ptypy/accelerate/cuda_pycuda/dependencies.yml ./dependencies.yml
COPY ./cufft/dependencies.yml ./dependencies_cufft.yml
RUN micromamba install cuda-version=${CUDAVERSION} && \
    micromamba env update -n base -f dependencies.yml && \
    micromamba env update -n base -f dependencies_cufft.yml && \
    micromamba clean -y --all --force-pkgs-dirs

# Pull from MPI build and install accelerate/cupy dependencies
FROM mpi AS cupy
ARG CUDAVERSION
COPY ./ptypy/accelerate/cuda_cupy/dependencies.yml ./dependencies.yml
COPY ./cufft/dependencies.yml ./dependencies_cufft.yml
RUN micromamba install cuda-version=${CUDAVERSION} && \
    micromamba env update -n base -f dependencies.yml && \
    micromamba env update -n base -f dependencies_cufft.yml && \
    micromamba clean -y --all --force-pkgs-dirs

# Pull from platform specific image and install ptypy 
FROM ${PLATFORM} AS build
COPY . .
RUN micromamba run pip install .

# For core/full build, no post processing needed
FROM build AS core-post
FROM build AS full-post

# For pycuda build, install filtered cufft
FROM build AS pycuda-post
RUN pip install ./cufft

# For pycuda build, install filtered cufft
FROM build AS cupy-post
RUN pip install ./cufft

# Platform specific runtime container
FROM ${PLATFORM}-post AS runtime

# Run PtyPy run script as entrypoint
ENTRYPOINT ["ptypy.cli"]
