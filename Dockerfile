# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE=ghcr.io/imec-int/hespas:jax
ARG URLREF_ASTRA_SIM=https://github.com/astra-sim/astra-sim.git
ARG ASTRASIM_COMMIT=28f18ea0d58a96fd812e49b71bdac330a1e3c965
ARG SRC_PATH_ASTRA_SIM=/opt/astra-sim

FROM ${BASE_IMAGE}
ARG URLREF_ASTRA_SIM
ARG SRC_PATH_ASTRA_SIM

RUN rm -rf /usr/local/bin/protoc
RUN apt update
RUN apt install -y cmake bc
RUN apt install -y protobuf-compiler libprotobuf-dev

RUN git clone ${URLREF_ASTRA_SIM} ${SRC_PATH_ASTRA_SIM} && \
    cd ${SRC_PATH_ASTRA_SIM} && \
    git checkout ${ASTRASIM_COMMIT}

RUN cd ${SRC_PATH_ASTRA_SIM} && \
    git submodule update --init --recursive 

# cxxopt.patch: add missing #include <cstdint>
RUN <<"EOF" bash -ex -o pipefail
CXXOPT_FILE="${SRC_PATH_ASTRA_SIM}/extern/helper/cxxopts/cxxopts.hpp"
if [ -f "${CXXOPT_FILE}" ] && ! grep -q '<cstdint>' "${CXXOPT_FILE}"; then
    sed -i 's|#include <cstring>|#include <cstring>\n#include <cstdint>|' "${CXXOPT_FILE}"
    echo "Applied cxxopt patch" >&2
fi
EOF

RUN cd ${SRC_PATH_ASTRA_SIM}/build/astra_analytical && \
    ./build.sh

RUN cp ${SRC_PATH_ASTRA_SIM}/build/astra_analytical/build/bin/* /usr/local/bin/

COPY --chmod=755 . /opt/hespas

WORKDIR /opt/hespas

RUN  pip install -I /opt/hespas
