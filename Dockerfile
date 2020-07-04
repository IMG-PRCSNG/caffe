FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 AS nvcaffe-env

ARG python_version_prefix
ARG python_version
ARG python_version_full

ENV PREFIX=/opt/install
ENV LD_LIBRARY_PATH=${PREFIX}/lib:${LD_LIBRARY_PATH} \
    PATH=${PREFIX}/bin:${PATH} \
    PYTHONPATH=${PREFIX}/python \
    PYTHON_VERSION=${python_version_full}

RUN \
    buildDeps="ca-certificates \
               software-properties-common \
               apt-transport-https \
               curl \
               git \
               bzip2 \
               gnupg \
               build-essential \
               make \
               pkg-config \
               libssl-dev \
               libbz2-dev \
               libreadline-dev \
               libsqlite3-dev \
               libcurl4-openssl-dev \
               libncurses5-dev \
               libncursesw5-dev \
               python${python_version_prefix}-distutils \
               python${python_version}-dev \
               zlib1g-dev" && \
    rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get -yqq update && \
    apt-get install -yq --no-install-recommends ${buildDeps} && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    curl -L "https://bootstrap.pypa.io/get-pip.py" | python${python_version} && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /scratch

## Install CMAKE
RUN \
    curl -sLO "https://github.com/Kitware/CMake/releases/download/v3.14.1/cmake-3.14.1-Linux-x86_64.sh" && \
    /bin/bash cmake-3.14.1-Linux-x86_64.sh \
        --prefix=/usr/local \
        --exclude-subdir \
        --skip-license

RUN caffe_deps="libopenblas-dev \
                libboost-all-dev \
                libgflags-dev \
                libgoogle-glog-dev \
                libhdf5-dev \
                libhdf5-serial-dev \
                libleveldb-dev \
                liblmdb-dev \
                libsnappy-dev \
                libopencv-dev \
                libprotobuf-dev \
                protobuf-compiler \
                libconfig++-dev \
                libturbojpeg0-dev" && \
    apt-get -yqq update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends ${caffe_deps} && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY ./python/requirements.txt .
RUN for req in $(cat requirements.txt); do pip${python_version} install $req; done

LABEL "Version"="0.17.4" \
      "Description"="NVCaffe" \
      "Maintainer"="samthedevil.sp@gmail.com"

FROM nvcaffe-env as nvcaffe-dev

WORKDIR /caffe
COPY ./build ./build
COPY . .
RUN build.sh

FROM scratch AS nvcaffe-package
COPY --from=nvcaffe-dev /opt/install /opt/install 
