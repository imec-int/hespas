#!/bin/sh
# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd -P)"
RESULTS_FILE="${SCRIPT_DIR}/results.csv"

EXPS_NAME="$(basename "${SCRIPT_DIR}")"
RUN_SCRIPT_NAME="run_exp.sh"
echo "Running all ${EXPS_NAME} paper experiments"
cd "${SCRIPT_DIR}"
INTERESTING_HEADERS="system workload runtime_s"
HEADER=

for i in *
do
    cd "${SCRIPT_DIR}"
    if [ -f "${SCRIPT_DIR}/$i" ]
    then
        continue
    fi
    SOURCE="$(basename $i)"
    EXP_DIR="${SCRIPT_DIR}/${SOURCE}"
    EXP_SCRIPT="${EXP_DIR}/${RUN_SCRIPT_NAME}"
    EXP_RESULTS="${EXP_DIR}/runs/results.csv"

    if [ ! -f "${EXP_SCRIPT}" ]
    then
        echo "Skipping ${SOURCE} (no ${RUN_SCRIPT_NAME})"
        continue
    fi
    cd "${EXP_DIR}"
    echo "Running experiment for ${SOURCE}"
    ./${RUN_SCRIPT_NAME} || exit 1
    cd "${SCRIPT_DIR}"
    if [ ! -f "${EXP_RESULTS}" ]
    then
        echo "Cannot find results for ${SOURCE}"
        exit 1
    fi

    echo "Collecting results for ${SOURCE}"
    WANTED_FIELDS=
    FIELD_NO=1
    for col in $(head -1 "${EXP_RESULTS}" | tr ',' ' ')
    do
        if [ -n "$(echo "${INTERESTING_HEADERS}" | grep "${col}")" ]
        then
            WANTED_FIELDS="${WANTED_FIELDS} ${FIELD_NO}"
        fi
        FIELD_NO="$(expr "${FIELD_NO}" "+" "1")"
    done
    WANTED_FIELDS="$(echo "${WANTED_FIELDS}" | tr -s ' ' ',' | sed 's/^,//')"

    if [ -z "${HEADER}" ]
    then
        HEADER="$(head -1 ${EXP_RESULTS} | cut -f "${WANTED_FIELDS}" -d ','),source"
        echo "${HEADER}" > "${RESULTS_FILE}"
    fi
    tail -n+2 "${EXP_RESULTS}" | cut -f "${WANTED_FIELDS}" -d ',' | while IFS= read -r line; do
        echo "${line},${SOURCE}"
    done >> "${RESULTS_FILE}"
done

# Append reference results
REFS_DIR="$(realpath "${SCRIPT_DIR}/../refs")"
REF_FILE="${REFS_DIR}/llama3_train.csv"
if [ -f "${REF_FILE}" ]; then
    echo "Appending reference results from ${REF_FILE}"
    tail -n+2 "${REF_FILE}" >> "${RESULTS_FILE}"
fi

echo "All experiments complete"
