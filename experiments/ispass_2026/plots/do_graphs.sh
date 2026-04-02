#!/bin/sh
# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT

set -ex

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
cd "${SCRIPT_DIR}"

if [ -f "llama3_train.csv" ]; then
    ./graph.py llama3_train.csv
fi

if [ -f "tpuv3_llama3.csv" ]; then
    ./graph.py tpuv3_llama3.csv
fi

if [ -f "resnet_train.csv" ]; then
    ./graph.py resnet_train.csv 1 1 True
fi
