#!/bin/sh

SCRIPT_PATH="$(realpath "$0")"
SCRIPT_DIR="$(realpath "$(dirname "$0")")"
if [ -f "${SCRIPT_DIR}/common.sh" ]
then
    . "${SCRIPT_DIR}/common.sh"
fi

INPUT_FILE=
OUTPUT_FILE=
GPU_SPEC=
RUN_DOCKER="no"
DISABLED_PASSES="async-collective-conversion"

usage()
{
    echo "$0: XLA optimiser script"
    echo "Usage: $0 -i INPUT_FILE -o OUTPUT_FILE -g GPU_SPEC_FILE [-d DISABLED_PASSES_LIST] [--docker]"
    echo "-d: Command seperated list of disabled passes. By default: '${DISABLED_PASSES}'"
    echo "--docker: Run the opt in the docker"
}

check_param()
{
    PARAM="$1"
    shift
    METAVAR="$1"
    shift
    INPUT_VAR="$1"
    shift
    if [ -z "${INPUT_VAR}" ]
    then
        echo "${PARAM} ${METAVAR} is required"
        usage
        exit 1
    fi
}

check_filepath()
{
    PARAM="$1"
    shift
    METAVAR="$1"
    shift
    INPUT_VAR="$1"
    shift
    if [ "$#" -ne 0 ]
    then
        IS_OUTPUT="$1"
        shift
    fi
    check_param "${PARAM}" "${METAVAR}" "${INPUT_VAR}"
    if [ "${IS_OUTPUT}" != "output" ] && [ ! -f "${INPUT_VAR}" ]
    then
        echo "Cannot find ${METAVAR} '${INPUT_VAR}'"
        exit 1
    elif [ "${IS_OUTPUT}" = "output" ] && [ ! -d "$(dirname "${INPUT_VAR}")" ]
    then
        echo "Cannot find output path for ${METAVAR} '${INPUT_VAR}'"
        exit 1
    fi
}

process_args()
{
    while [ "$#" -gt 0 ]
    do
        case "$1" in 
            -h|--help)
                usage
                exit 0
                ;;
            -i)
                shift
                INPUT_FILE="$1"
                ;;
            -o)
                shift
                OUTPUT_FILE="$1"
                ;;
            -g)
                shift
                GPU_SPEC="$1"
                ;;
            -d)
                shift
                DISABLED_PASSES="$1"
                ;;
            --docker)
                RUN_DOCKER="yes"
                ;;
            *)
                echo "Unknown argument '$1'"
                echo ""
                usage
                exit 1
                ;;
        esac
        shift
    done

    check_filepath "-i" "INPUT_FILE" "${INPUT_FILE}"
    check_filepath "-o" "OUTPUT_FILE" "${OUTPUT_FILE}" "output"
    check_filepath "-g" "GPU_SPEC_FILE" "${GPU_SPEC}"
}

run_opt()
{
    XLA_TRANSLATE="$1"
    shift
    XLA_HLO_OPT="$1"
    shift
    INPUT_FILE="$(realpath "$1")"
    shift
    OUTPUT_FILE="$(realpath -m "$1")"
    shift
    GPU_SPEC="$(realpath "$1")"
    shift
    DISABLED_PASSES="$1"
    shift

    UNOP_HLO_TEMP="${INPUT_FILE}.unop.hlo"
    OP_HLO_TEMP="${INPUT_FILE}.op.hlo"

    rm -f "${UNOP_HLO_TEMP}" "${OP_HLO_TEMP}"

    "${XLA_TRANSLATE}" \
        --stablehlo-to-hlo-text \
        -o="${UNOP_HLO_TEMP}" \
        "${INPUT_FILE}"

    "${XLA_HLO_OPT}" \
        --platform=gpu \
        --xla_gpu_target_config_filename="${GPU_SPEC}" \
        --xla_disable_hlo_passes="${DISABLED_PASSES}" \
    	--xla_gpu_disable_async_collectives="ALLCOLLECTIVES" \
    	--xla_gpu_enable_latency_hiding_scheduler=false \
        --xla_gpu_autotune_level=0 \
    	--o="${OP_HLO_TEMP}" \
        "${UNOP_HLO_TEMP}"

    "${XLA_TRANSLATE}" \
        --emit-stablehlo \
        --hlo-text-to-stablehlo \
        --hlo-flatten-computation-args-result \
        --hlo-import-all-computations \
        --enable-cse-in-irtranslator \
        -o="${OUTPUT_FILE}" \
        "${OP_HLO_TEMP}"

    rm "${UNOP_HLO_TEMP}" "${OP_HLO_TEMP}"
}

process_args "$@"
set -e

run_opt "$(which xla-translate)" "$(which hlo-opt)" "${INPUT_FILE}" "${OUTPUT_FILE}" "${GPU_SPEC}" "${DISABLED_PASSES}"