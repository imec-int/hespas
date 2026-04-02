#!/bin/sh
# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
#
# Merge experiment results with reference data and generate plots.

set -e

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
ISPASS_DIR="$(realpath "${SCRIPT_DIR}/..")"
REFS_DIR="${ISPASS_DIR}/refs"

# merge_and_plot output_csv experiment_results reference_data [system_filter] [workload_filter]
merge_and_plot() {
    out="$1"
    results="$2"
    ref="$3"
    sys_filter="${4:-}"
    wl_filter="${5:-}"

    if [ ! -f "${results}" ]; then
        echo "Warning: Results file not found: ${results}, skipping" >&2
        return
    fi

    # Start with header
    head -1 "${results}" > "${out}"

    # Append experiment results, optionally filtered
    if [ -n "${sys_filter}" ] || [ -n "${wl_filter}" ]; then
        tail -n+2 "${results}" | while IFS= read -r line; do
            skip=false
            if [ -n "${sys_filter}" ]; then
                echo "${line}" | grep -q "${sys_filter}" || skip=true
            fi
            if [ -n "${wl_filter}" ] && [ "${skip}" = "false" ]; then
                echo "${line}" | grep -q "${wl_filter}" || skip=true
            fi
            if [ "${skip}" = "false" ]; then
                echo "${line}"
            fi
        done >> "${out}"
    else
        tail -n+2 "${results}" >> "${out}"
    fi

    # Append reference data (skip header) if available
    if [ -f "${ref}" ]; then
        tail -n+2 "${ref}" >> "${out}"
        echo "Merged ${results} + ${ref} → ${out}"
    else
        echo "Warning: Reference file not found: ${ref}, plotting without reference" >&2
    fi
}

merge_and_plot "${SCRIPT_DIR}/llama3_train.csv" \
    "${ISPASS_DIR}/llama3/results.csv" \
    "${REFS_DIR}/llama3_train.csv"

merge_and_plot "${SCRIPT_DIR}/resnet_train.csv" \
    "${ISPASS_DIR}/resnet/results.csv" \
    "${REFS_DIR}/resnet_train.csv" \
    "A100" "_train"

merge_and_plot "${SCRIPT_DIR}/tpuv3_llama3.csv" \
    "${ISPASS_DIR}/tpu/results.csv" \
    "${REFS_DIR}/tpuv3_llama3.csv"

echo "Generating graphs"
cd "${SCRIPT_DIR}"
./do_graphs.sh

echo "Plotting complete — output in ${SCRIPT_DIR}"
