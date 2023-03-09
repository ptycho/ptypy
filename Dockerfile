# Set global build arguments
ARG PYTHON_VERSION=3.9
ARG PLATFORM=pycuda

# Pull mambaforge image and install accelerate dependencies
FROM condaforge/mambaforge as accelerate

ARG PYTHON_VERSION=3.9
ARG PLATFORM=pycuda
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y libxml2
COPY ./ptypy/accelerate/cuda_${PLATFORM}/dependencies.yml ./dependencies.yml
RUN sed -i "s/python=3.9/python=${PYTHON_VERSION}/" dependencies.yml
RUN mamba env update -n base -f dependencies.yml && mamba install -y -n base pytest

# Pull from miniconda image and install core dependencies
FROM continuumio/miniconda3 as core

ARG PYTHON_VERSION=3.9
COPY ./dependencies_core.yml ./dependencies.yml
RUN sed -i "s/python=3.9/python=${PYTHON_VERSION}/" dependencies.yml
RUN conda env update -n base -f dependencies.yml && conda install -y -n base pytest

# Pull from existing image with ptypy dependencies and set up testing
FROM localhost/ptypy_${PLATFORM}_py${PYTHON_VERSION}_devel:latest as runtime
ENV WORKDIR=/ptypy
WORKDIR ${WORKDIR}
COPY . ${WORKDIR}
RUN cd ${WORKDIR} && pip install .
RUN cd ${WORKDIR}/cufft && pip install .