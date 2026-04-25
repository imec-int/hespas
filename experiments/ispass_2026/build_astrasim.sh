#!/bin/sh
# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
#
# Clone, patch, and build the specific version of Astra-sim required by the
# paper experiments. Prints the bin directory path to stdout on success.

set -e

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
ASTRASIM_SRC_DIR="${SCRIPT_DIR}/.astrasim"
ASTRASIM_COMMIT="831eca84b41b1a3d9c1c527d1153c254026ab029"
ASTRASIM_REPO="https://github.com/astra-sim/astra-sim.git"
ASTRASIM_BIN_DIR="${ASTRASIM_SRC_DIR}/build/astra_analytical/build/bin"

# Check build dependencies
for cmd in git cmake protoc bc; do
    if ! command -v "${cmd}" > /dev/null 2>&1; then
        echo "Error: Required command '${cmd}' not found. Please install it." >&2
        exit 1
    fi
done

if [ -x "${ASTRASIM_BIN_DIR}/AstraSim_Analytical_Congestion_Unaware" ]; then
    echo "Astra-sim already built at ${ASTRASIM_BIN_DIR}" >&2
    echo "${ASTRASIM_BIN_DIR}"
    exit 0
fi

echo "=== Building Astra-sim (commit ${ASTRASIM_COMMIT}) ===" >&2

if [ ! -d "${ASTRASIM_SRC_DIR}" ]; then
    echo "Cloning Astra-sim..." >&2
    git clone --recursive "${ASTRASIM_REPO}" "${ASTRASIM_SRC_DIR}"
fi

cd "${ASTRASIM_SRC_DIR}"
git checkout "${ASTRASIM_COMMIT}"
git submodule update --init --recursive

# cxxopt.patch: add missing #include <cstdint>
CXXOPT_FILE="${ASTRASIM_SRC_DIR}/extern/helper/cxxopts/cxxopts.hpp"
if [ -f "${CXXOPT_FILE}" ] && ! grep -q '<cstdint>' "${CXXOPT_FILE}"; then
    sed -i 's|#include <cstring>|#include <cstring>\n#include <cstdint>|' "${CXXOPT_FILE}"
    echo "Applied cxxopt patch" >&2
fi

# macos_build.patch: portable nproc in build scripts
for BUILD_SCRIPT in \
    "${ASTRASIM_SRC_DIR}/build/astra_analytical/build.sh" \
    "${ASTRASIM_SRC_DIR}/build/astra_htsim/build.sh" \
    "${ASTRASIM_SRC_DIR}/build/astra_ns3/build.sh"
do
    if [ -f "${BUILD_SCRIPT}" ] && grep -q 'NUM_THREADS=$(nproc)' "${BUILD_SCRIPT}"; then
        sed -i '/^CHAKRA_ET_DIR=.*$/a\
if [ -z "${NPROC}" ]\
then\
    NPROC="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "2")"\
fi' "${BUILD_SCRIPT}"
        sed -i 's|NUM_THREADS=$(nproc)|NUM_THREADS="${NPROC}"|' "${BUILD_SCRIPT}"
        echo "Applied nproc patch to $(basename "${BUILD_SCRIPT}")" >&2
    fi
done

echo "Building Astra-sim analytical backend..." >&2
cd "${ASTRASIM_SRC_DIR}/build/astra_analytical"
./build.sh

if [ ! -x "${ASTRASIM_BIN_DIR}/AstraSim_Analytical_Congestion_Unaware" ]; then
    echo "Error: Astra-sim build failed - binary not found at ${ASTRASIM_BIN_DIR}" >&2
    exit 1
fi

echo "=== Astra-sim built successfully ===" >&2
echo "${ASTRASIM_BIN_DIR}"
