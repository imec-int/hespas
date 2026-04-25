#!/bin/sh
# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
#
# Clone and build the Collective API required by the paper experiments.

set -ex

# Check if running in a Python virtual environment
if [ -z "${VIRTUAL_ENV}" ] && [ -z "${CONDA_PREFIX}" ]; then
    echo "Error: Not running in a Python virtual environment (venv or conda)." >&2
    echo "Please activate a virtual environment first:" >&2
    echo "  python -m venv /path/to/venv && source /path/to/venv/bin/activate" >&2
    echo "  or" >&2
    echo "  conda activate myenv" >&2
    exit 1
fi

# Check build dependencies
for cmd in git; do
    if ! command -v "${cmd}" > /dev/null 2>&1; then
        echo "Error: Required command '${cmd}' not found. Please install it." >&2
        exit 1
    fi
done

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
COLLAPI_SRC_DIR="${SCRIPT_DIR}/.collectiveapi"
COLLAPI_REPO="git@github.com:astra-sim/collectiveapi"

# Check build dependencies
echo "=== Building Collective API ===" >&2

echo "Cloning Collective API..." >&2
git clone "${COLLAPI_REPO}" "${COLLAPI_SRC_DIR}"

cd "${COLLAPI_SRC_DIR}"

echo "Initializing and updating submodules..." >&2
git submodule init
git submodule update

echo "Installing msccl-tools..." >&2
cd "${COLLAPI_SRC_DIR}/msccl-tools"
pip install .

echo "=== Collective API built successfully ===" >&2
