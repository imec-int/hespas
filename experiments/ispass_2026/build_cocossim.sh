#!/bin/sh
# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
#
# Download, patch, and build CoCoSim (perf_model) required by TPU experiments.
# Prints the bin directory path to stdout on success.

set -e

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
COCOSSIM_SRC_DIR="${SCRIPT_DIR}/.cocossim"
COCOSSIM_BUILD_DIR="${COCOSSIM_SRC_DIR}/build"
COCOSSIM_BIN_DIR="${COCOSSIM_BUILD_DIR}"
COCOSSIM_ZIP_URL="https://ndownloader.figshare.com/files/53060471"
DRAMSIM_REPO="https://github.com/umd-memsys/DRAMsim3.git"

for cmd in cmake curl unzip git; do
    if ! command -v "${cmd}" > /dev/null 2>&1; then
        echo "Error: Required command '${cmd}' not found. Please install it." >&2
        exit 1
    fi
done

if [ -x "${COCOSSIM_BIN_DIR}/perf_model" ]; then
    echo "CoCoSim already built at ${COCOSSIM_BIN_DIR}" >&2
    echo "${COCOSSIM_BIN_DIR}"
    exit 0
fi

echo "=== Building CoCoSim ===" >&2

if [ ! -d "${COCOSSIM_SRC_DIR}" ]; then
    echo "Downloading CoCoSim..." >&2
    COCOSSIM_ZIP="${SCRIPT_DIR}/.cocossim.zip"
    curl -L "${COCOSSIM_ZIP_URL}" -o "${COCOSSIM_ZIP}"
    mkdir -p "${COCOSSIM_SRC_DIR}"
    unzip -o "${COCOSSIM_ZIP}" -d "${COCOSSIM_SRC_DIR}"
    rm -f "${COCOSSIM_ZIP}"

    # The zip extracts into a perf_model subdirectory
    if [ -d "${COCOSSIM_SRC_DIR}/perf_model" ]; then
        rm -rf "${COCOSSIM_SRC_DIR}/__MACOSX"
        mv "${COCOSSIM_SRC_DIR}/perf_model"/* "${COCOSSIM_SRC_DIR}"
        rm -rf "${COCOSSIM_SRC_DIR}/perf_model"
    fi
fi

# Clone DRAMsim3 if not present
if [ ! -d "${COCOSSIM_SRC_DIR}/dramsim3" ]; then
    echo "Cloning DRAMsim3..." >&2
    git clone "${DRAMSIM_REPO}" "${COCOSSIM_SRC_DIR}/dramsim3"
fi

cd "${COCOSSIM_SRC_DIR}"

# Patch: adjust n_mxus and n_vpus for TPUv3
GLOBAL_H="${COCOSSIM_SRC_DIR}/include/global.h"
if [ -f "${GLOBAL_H}" ] && grep -q 'n_mxus = 4' "${GLOBAL_H}"; then
    sed -i 's/const int n_mxus = 4;/const int n_mxus = 2;/' "${GLOBAL_H}"
    sed -i 's/const int n_vpus = 4;/const int n_vpus = 1;/' "${GLOBAL_H}"
    echo "Applied CoCoSim num_eles patch" >&2
fi

# Patch: fix DRAMsim3 cmake minimum version
DRAMSIM_CMAKE="${COCOSSIM_SRC_DIR}/dramsim3/CMakeLists.txt"
if [ -f "${DRAMSIM_CMAKE}" ] && grep -q 'cmake_minimum_required(VERSION 3.0.0)' "${DRAMSIM_CMAKE}"; then
    sed -i 's/cmake_minimum_required(VERSION 3.0.0)/cmake_minimum_required(VERSION 3.6.0)/' "${DRAMSIM_CMAKE}"
    echo "Applied DRAMsim3 cmake patch" >&2
fi

echo "Building CoCoSim..." >&2
cmake -S "${COCOSSIM_SRC_DIR}" -B "${COCOSSIM_BUILD_DIR}" -DUSE_VCD=ON
cmake --build "${COCOSSIM_BUILD_DIR}"

if [ ! -x "${COCOSSIM_BIN_DIR}/perf_model" ]; then
    echo "Error: CoCoSim build failed - perf_model binary not found at ${COCOSSIM_BIN_DIR}" >&2
    exit 1
fi

echo "=== CoCoSim built successfully ===" >&2
echo "${COCOSSIM_BIN_DIR}"
