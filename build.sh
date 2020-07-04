#!/bin/bash

set -euxo pipefail

MAKEFLAGS="-j$(nproc)"

# Add this option to use your own opencv
# -DOpenCV_DIR="${PREFIX}/share/OpenCV" \
cmake \
    -S . \
    -B build \
    -DCMAKE_BUILD_TYPE="Release" \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DBUILD_docs=0 \
    -DBUILD_matlab=0 \
    -DBUILD_python=1 \
    -DBUILD_python_layer=1 \
    -DBLAS=Open \
    -DUSE_CUDNN=ON \
    -DUSE_LEVELDB=OFF \
    -DUSE_LMDB=OFF \
    -DUSE_NCCL=OFF \
    -DCUDA_ARCH_NAME=Manual \
    -DCUDA_ARCH_PTX="37 50 60 70 75" \
    -DCUDA_ARCH_BIN="37 50 52 60 61 70 75" \
    -Dpython_version=3 \
    -DUSE_OPENCV=ON && \
    make ${MAKEFLAGS} -C build && \
    make ${MAKEFLAGS} -C build install
