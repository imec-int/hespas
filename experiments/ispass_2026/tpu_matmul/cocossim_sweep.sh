#!/bin/sh
# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd -P)"
PAPER_DIR="$(realpath "${SCRIPT_DIR}/..")"

COCOSSIM_BIN_DIR="$("${PAPER_DIR}/build_cocossim.sh")"
export PATH="${COCOSSIM_BIN_DIR}:${PATH}"
export LD_LIBRARY_PATH="${PAPER_DIR}/.cocossim/dramsim3:${LD_LIBRARY_PATH}"

PERF_MODEL_DIR="$(dirname "$(which perf_model)")"
OUT_FILE="out.txt"

cd "$PERF_MODEL_DIR"
echo "Matmul 32 32 32" > layers.txt
./perf_model -c 4 -sa_sz 128 -vu_sz 1024 -ws 1 -f 1.5 -i layers.txt -o ${OUT_FILE}
cat ${OUT_FILE}

echo "Matmul 64 64 64" > layers.txt
./perf_model -c 4 -sa_sz 128 -vu_sz 1024 -ws 1 -f 1.5 -i layers.txt -o ${OUT_FILE}
cat ${OUT_FILE}

echo "Matmul 128 128 128" > layers.txt
./perf_model -c 4 -sa_sz 128 -vu_sz 1024 -ws 1 -f 1.5 -i layers.txt -o ${OUT_FILE}
cat ${OUT_FILE}

echo "Matmul 256 256 256" > layers.txt
./perf_model -c 4 -sa_sz 128 -vu_sz 1024 -ws 1 -f 1.5 -i layers.txt -o ${OUT_FILE}
cat ${OUT_FILE}

echo "Matmul 512 512 512" > layers.txt
./perf_model -c 4 -sa_sz 128 -vu_sz 1024 -ws 1 -f 1.5 -i layers.txt -o ${OUT_FILE}
cat ${OUT_FILE}

echo "Matmul 1024 1024 1024" > layers.txt
./perf_model -c 4 -sa_sz 128 -vu_sz 1024 -ws 1 -f 1.5 -i layers.txt -o ${OUT_FILE}
cat ${OUT_FILE}

echo "Matmul 2048 2048 2048" > layers.txt
./perf_model -c 4 -sa_sz 128 -vu_sz 1024 -ws 1 -f 1.5 -i layers.txt -o ${OUT_FILE}
cat ${OUT_FILE}

echo "Matmul 4096 4096 4096" > layers.txt
./perf_model -c 4 -sa_sz 128 -vu_sz 1024 -ws 1 -f 1.5 -i layers.txt -o ${OUT_FILE}
cat ${OUT_FILE}
