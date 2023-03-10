# Set global build arguments
ARG PYTHON_VERSION=3.9
ARG PLATFORM=pycuda

# Pull from miniconda image and install core dependencies
FROM continuumio/miniconda3 as core

ARG PYTHON_VERSION=3.9
COPY ./dependencies_core.yml ./dependencies.yml
RUN sed -i "s/python=3.9/python=${PYTHON_VERSION}/" dependencies.yml
RUN conda env update -n base -f dependencies.yml && conda install -y -n base pytest

# Pull from miniconda image and install full dependencies
FROM condaforge/mambaforge as full

ARG PYTHON_VERSION=3.9
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y libxml2 ssh
COPY ./dependencies_full.yml ./dependencies.yml
RUN sed -i "s/python=3.9/python=${PYTHON_VERSION}/" dependencies.yml
RUN mamba env update -n base -f dependencies.yml && mamba install -y -n base pytest

# Pull mambaforge image and install accelerate/pycuda dependencies
FROM condaforge/mambaforge as pycuda

ARG PYTHON_VERSION=3.9
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y libxml2 ssh
COPY ./ptypy/accelerate/cuda_pycuda/dependencies.yml ./dependencies.yml
RUN sed -i "s/python=3.9/python=${PYTHON_VERSION}/" dependencies.yml
RUN mamba env update -n base -f dependencies.yml && mamba install -y -n base pytest pybind11

# Pull mambaforge image and install accelerate/cupy dependencies
FROM condaforge/mambaforge as cupy

ARG PYTHON_VERSION=3.9
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y libxml2 ssh
COPY ./ptypy/accelerate/cuda_cupy/dependencies.yml ./dependencies.yml
RUN sed -i "s/python=3.9/python=${PYTHON_VERSION}/" dependencies.yml
RUN mamba env update -n base -f dependencies.yml && mamba install -y -n base pytest pybind11

# Pull from existing image with ptypy dependencies and set up testing
FROM ${PLATFORM} as runtime
COPY ./ ./
RUN pip install .
RUN if [ "$PLATFORM" = "pycuda" ] || [ "$PLATFORM" = "cupy" ] ; then pip install ./cufft ; fi