# Set global build arguments
ARG PLATFORM=cupy
n
# Pull from miniconda image and install core dependencies
FROM continuumio/miniconda3 as core

# Basic conda installation
COPY ./dependencies_core.yml ./dependencies.yml
RUN conda env update -n base -f dependencies.yml && conda install -y -n base pytest

# Pull from miniconda image and install full dependencies
FROM condaforge/mambaforge as full

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y libxml2 ssh
COPY ./dependencies_full.yml ./dependencies.yml
RUN mamba env update -n base -f dependencies.yml && mamba install -y -n base pytest

# Pull mambaforge image and install accelerate/pycuda dependencies
FROM condaforge/mambaforge as pycuda

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y libxml2 ssh
COPY ./ptypy/accelerate/cuda_pycuda/dependencies.yml ./dependencies.yml
COPY ./cufft/dependencies.yml ./dependencies_cufft.yml
RUN mamba env update -n base -f dependencies.yml && mamba env update -n base -f dependencies_cufft.yml

# Pull mambaforge image and install accelerate/cupy dependencies
FROM condaforge/mambaforge as cupy

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y libxml2 ssh
COPY ./ptypy/accelerate/cuda_cupy/dependencies.yml ./dependencies.yml
COPY ./cufft/dependencies.yml ./dependencies_cufft.yml
RUN mamba env update -n base -f dependencies.yml && mamba env update -n base -f dependencies_cufft.yml

# Pull from existing image with ptypy dependencies and set up testing
FROM ${PLATFORM} as runtime
COPY ./ ./
RUN pip install .
RUN if [ "$PLATFORM" = "pycuda" ] || [ "$PLATFORM" = "cupy" ] ; then pip install ./cufft ; fi

# Run PtyPy CLI as entrypoint
ENTRYPOINT ["ptypy.cli"]
