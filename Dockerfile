# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE=ghcr.io/imec-int/hespas:jax
ARG URLREF_ASTRA_SIM=git@github.imec.be:HeSPaS/astra-sim.git
ARG SRC_PATH_ASTRA_SIM=/opt/astra-sim

FROM ${BASE_IMAGE}
ARG URLREF_ASTRA_SIM
ARG SRC_PATH_ASTRA_SIM

RUN rm -rf /usr/local/bin/protoc
RUN apt update
RUN apt install -y cmake bc
RUN apt install -y protobuf-compiler libprotobuf-dev

RUN --mount=type=ssh \
    --mount=type=secret,id=SSH_KNOWN_HOSTS,target=/root/.ssh/known_hosts \
    git clone ${URLREF_ASTRA_SIM} ${SRC_PATH_ASTRA_SIM}
RUN cd ${SRC_PATH_ASTRA_SIM} && \
    git submodule update --init --recursive 
RUN cd ${SRC_PATH_ASTRA_SIM}/build/astra_analytical && \
    ./build.sh
RUN cp ${SRC_PATH_ASTRA_SIM}/build/astra_analytical/build/bin/* /usr/local/bin/

COPY --chmod=755 . /opt/hespas

WORKDIR /opt/hespas

RUN  pip install -I /opt/hespas
