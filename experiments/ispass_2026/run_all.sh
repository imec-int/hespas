#!/bin/sh
# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd -P)"
PAPER_DIR="${SCRIPT_DIR}"

echo "=== Building dependencies ==="
ASTRASIM_BIN_DIR="$("${PAPER_DIR}/build_astrasim.sh")"
COCOSSIM_BIN_DIR="$("${PAPER_DIR}/build_cocossim.sh")"
export PATH="${ASTRASIM_BIN_DIR}:${COCOSSIM_BIN_DIR}:${PATH}"
export LD_LIBRARY_PATH="${PAPER_DIR}/.cocossim/dramsim3:${LD_LIBRARY_PATH}"

echo "=== Running llama3 experiments ==="
"${PAPER_DIR}/llama3/run_exps.sh"

echo "=== Running resnet experiments ==="
"${PAPER_DIR}/resnet/run_exps.sh"

echo "=== Running TPU experiments ==="
"${PAPER_DIR}/tpu/run_exps.sh"

echo "=== Generating plots ==="
"${PAPER_DIR}/plots/plot.sh"

echo "=== Done ==="
