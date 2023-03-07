# This is the build image
FROM registry.access.redhat.com/ubi8 as build
USER root

# Install dependencies
RUN yum -y update && yum -y upgrade && yum clean all \
    && yum install -y nss-pam-ldapd mesa-libGL tree \
    && yum install -y openssh-server openssh-clients \
    && yum install -y wget \
    && sed -i 's/sss/ldap/g' /etc/nsswitch.conf \
    && yum -y install curl bzip2 \
    && yum -y install git \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local/ \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3 conda-build \
    && conda install -y conda-pack \
    && conda update -y conda \
    && conda clean --all --yes

# Set build arguments and environment variables
ARG PTYPY_BACKEND=pycuda
ENV CUDA_PATH=/usr/local/cuda \
    LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH \
    BACKEND=$PTYPY_BACKEND

# Copy dependencies
COPY ./ptypy/accelerate/cuda_${BACKEND}/dependencies.yml ./dependencies.yml

# Conda env for PTYPY CUDA/PYCUDA backend
RUN conda env create -n ptypy_${BACKEND} -f dependencies.yml \
    && conda install -y -n ptypy_${BACKEND} -c conda-forge pytest

# Use conda-pack to create a standalone env in /venv
# RUN conda-pack -n ptypy_${BACKEND} -o /tmp/env.tar && \
#     mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
#     rm /tmp/env.tar

# # Put venv in same path it'll be in the final image
# RUN /venv/bin/conda-unpack

# # This is the runtime image
# FROM registry.access.redhat.com/ubi8 as runtime

# # Copy /venv from the build stage
# COPY --from=build /venv /venv

# # Copy ptypy tests
# COPY ./test ./test

#SHELL ["/bin/bash", "-c"]

# Code to run with container has started
#CMD source /venv/bin/activate python

# Make sure RUN commands use the conda env
#SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

# Activate conda environment
#RUN cd ptypy && pip install .

# Install filtered cufft
#RUN cd cufft && conda env update -n ptypy_pycuda --file dependencies.yml && pip install .

# Run this when container is started
#ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "base"]