#!/bin/sh
# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT

set -xe

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd -P)"
ROOT_DIR="$(realpath "${SCRIPT_DIR}/../../../../")"
LIB_DIR="${ROOT_DIR}/experiments/shlib"
WORKLOADS_DIR="${SCRIPT_DIR}/workloads"
WORKLOADS_URL="git@github.com:imec-int/hespas_workloads.git"
WORKLOADS_BRANCH="llama-3-4-xla"
RUNS_DIR="${SCRIPT_DIR}/runs"
CSV_RESULTS_FILE="${RUNS_DIR}/results.csv"
CONFIGS_DIR="${ROOT_DIR}/configs"
NODES_DIR="${CONFIGS_DIR}/nodes"
SYSTEMS_DIR="${CONFIGS_DIR}/systems"

PAPER_DIR="${ROOT_DIR}/experiments/ispass_2026"
ASTRASIM_BIN_DIR="$("${PAPER_DIR}/build_astrasim.sh")"
export PATH="${ASTRASIM_BIN_DIR}:${PATH}"

echo "Running llama3 4GPU experiments"
echo "Deleting run dir '${RUNS_DIR}'"
rm -rf "${RUNS_DIR}"
echo "Creating run dir '${RUNS_DIR}'"
mkdir -p "${RUNS_DIR}"

echo "Pulling workloads branch '${WORKLOADS_BRANCH}' to '${WORKLOADS_DIR}'"
rm -rf "${WORKLOADS_DIR}"
git clone "${WORKLOADS_URL}" -b "${WORKLOADS_BRANCH}" --single-branch "${WORKLOADS_DIR}"
echo "Pulling workloads complete"

echo "Initialising output file '${CSV_RESULTS_FILE}'"
echo "system,workload,runtime_ns,runtime_s,exposed_comms_ns,exposed_comms_percent,time_taken_s" > "${CSV_RESULTS_FILE}"

NPUS="4"
FIELD_OFFSET="2"
for CONFIG_NAME in A100_SXM_40GB H100_80GB H200_141GB B200_180GB
do
    SYSTEM_CONFIG="${SYSTEMS_DIR}/${CONFIG_NAME}_${NPUS}GPU"
    system_type="$(echo "${CONFIG_NAME}" | cut -f 1 -d '_')"
    SYSTEM_WORKLOAD_DIR="${WORKLOADS_DIR}/${system_type}"
    for i in $(ls "${SYSTEM_WORKLOAD_DIR}" | tr 'b' 'G' | tr 'm' 'M' | sort -h -t '-' -k${FIELD_OFFSET} | tr 'M' 'm' | tr 'G' 'b')
    do
        START_TIME="$(date '+%s')"
        WORKLOAD_DIR="$i"
        WORKLOAD_NAME="$(basename "${WORKLOAD_DIR}")"
        RUN_NAME="${WORKLOAD_NAME}"
        RUN_DIR="${RUNS_DIR}/${CONFIG_NAME}/${RUN_NAME}"
        echo "Running system '${CONFIG_NAME}' with workload '${WORKLOAD_NAME}'"
        echo "Creating run dir '${RUN_DIR}'"
        mkdir -p "${RUN_DIR}"
        CHAKRA_OUTPUT_DIR="${SYSTEM_WORKLOAD_DIR}/${WORKLOAD_DIR}"
        HESPAS_OUT="${CHAKRA_OUTPUT_DIR}/hespas.out"
        ASTRA_SIM_DIR="${RUN_DIR}/astrasim"
        ASTRA_SIM_OUT="${ASTRA_SIM_DIR}/run.out"

        echo "Running Astra-sim congestion-unaware to generate performance estimations"
        mkdir -p "${ASTRA_SIM_DIR}"
        cp "${SYSTEM_CONFIG}/astra-sim"/* "${ASTRA_SIM_DIR}"
        cd "${ASTRA_SIM_DIR}"
        AstraSim_Analytical_Congestion_Unaware \
            --workload-configuration="${CHAKRA_OUTPUT_DIR}/dev" \
            --system-configuration="${ASTRA_SIM_DIR}/system.json" \
            --remote-memory-configuration="${ASTRA_SIM_DIR}/remote_memory.json" \
            --network-configuration="${ASTRA_SIM_DIR}/network.yml" \
            --comm-group-configuration="${CHAKRA_OUTPUT_DIR}/comm_group.json" \
            2>&1 | tee "${ASTRA_SIM_OUT}"
        if ! grep -q "sys\[0\] finished" "${ASTRA_SIM_OUT}"; then
            echo "Error: Astra-sim failed. Check the output above for details."
            exit 1
        fi
        echo "Running Astra-sim complete"

        echo "Collecting Astra-sim statistics"
        RUNTIME="$(sed -n "s/.*sys\[0\]\s*finished\s*,\s*\([0-9]\+\).*/\1/p" < "${ASTRA_SIM_OUT}")"
        RUNTIME_S="$(echo "scale=3; ${RUNTIME} / 1000000000" | bc)"
        EXPOSED_COMMS_TIME="$(sed -n "s/.*sys\[0\]\s*finished\s*,\s[0-9]\+\s*cycles\s*,\s*exposed\s*communication\s*\([0-9]\+\).*/\1/p" < "${ASTRA_SIM_OUT}")"
        EXPOSED_COMMS_PERCENT="$(echo "scale=3; ${EXPOSED_COMMS_TIME} / ${RUNTIME}" | bc)"
        TIME_TAKEN_ASTRASIM="$(echo "scale=3;$(date '+%s')-${START_TIME}" | bc)"
        HESPAS_TIME_TAKEN="$(grep Took "${HESPAS_OUT}" | grep -o "[0-9]\+\.[0-9]\+")"
        TIME_TAKEN="$(echo "scale=3; ${TIME_TAKEN_ASTRASIM}+${HESPAS_TIME_TAKEN}" | bc)"
        echo "Writing statistics to file '${CSV_RESULTS_FILE}'"
        echo "${CONFIG_NAME},${WORKLOAD_NAME},${RUNTIME},${RUNTIME_S},${EXPOSED_COMMS_TIME},${EXPOSED_COMMS_PERCENT},${TIME_TAKEN}" >> "${CSV_RESULTS_FILE}"
        echo "Running system '${CONFIG_NAME}' with workload '${WORKLOAD_NAME}' complete"
    done
done

echo "All experiments complete"
