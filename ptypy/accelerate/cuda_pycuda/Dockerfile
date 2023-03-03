# This includes cuda devtools
FROM nvidia/cuda:12.0.1-devel-centos7
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
    && conda install -y python=3 conda-build\
    && conda update -y conda \
    && conda clean --all --yes

# Set environment variables
ENV CUDA_PATH=/usr/local/cuda \
    LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Clone PtyPy
RUN git clone https://github.com/ptycho/ptypy.git

# Conda env for PTYPY CUDA/PYCUDA backend
RUN conda env create -f ptypy/ptypy/accelerate/cuda_pycuda/dependencies.yml \
    && conda install -n ptypy_pycuda pytest

# Activate conda environment
RUN source activate /usr/local/envs/ptypy_pycuda \
    && cd ptypy && pip install --user .
