#!/bin/bash

# Mapping from filename patterns to node config directories
declare -A GPU_MAP
GPU_MAP["a100_sxm_80"]="A100_SXM_80GB"
GPU_MAP["a100_sxm_40"]="A100_SXM_40GB"
GPU_MAP["a100_pcie_80"]="A100_PCIe_80GB"
GPU_MAP["a100_pcie_40"]="A100_PCIe_40GB"
GPU_MAP["b200"]="B200_180GB"
GPU_MAP["h100_sxm"]="H100_80GB"
GPU_MAP["h100"]="H100_80GB"
GPU_MAP["h200"]="H200_141GB"
GPU_MAP["v100"]="V100_32GB"
GPU_MAP["a40"]="A40_48GB"

# Function to detect GPU on the current machine using nvidia-smi
detect_gpu_with_nvidia_smi() {
  if ! command -v nvidia-smi &> /dev/null; then
      echo "Error: nvidia-smi not found. Cannot detect GPU." >&2
      return 1
  fi
  
  # Get GPU name from nvidia-smi
  local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1)
  
  if [ -z "$gpu_name" ]; then
      echo "Error: Could not detect GPU with nvidia-smi" >&2
      return 1
  fi
  
  echo "Detected GPU: ${gpu_name}" >&2
  
  # Convert GPU name to lowercase for pattern matching
  local gpu_lower=$(echo "$gpu_name" | tr '[:upper:]' '[:lower:]')
  
  # Map nvidia-smi GPU name to config directory
  if [[ "$gpu_lower" == *"a100"*"sxm"* ]]; then
      # Detect memory size for A100 SXM
      local mem_size=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
      local mem_gb=$(echo "$mem_size" | awk '{print int($1/1024)}')
      if [ "$mem_gb" -ge 75 ]; then
          echo "A100_SXM_80GB"
      else
          echo "A100_SXM_40GB"
      fi
  elif [[ "$gpu_lower" == *"a100"*"pcie"* ]] || [[ "$gpu_lower" == *"a100-pcie"* ]]; then
      # Detect memory size for A100 PCIe
      local mem_size=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
      local mem_gb=$(echo "$mem_size" | awk '{print int($1/1024)}')
      if [ "$mem_gb" -ge 75 ]; then
          echo "A100_PCIe_80GB"
      else
          echo "A100_PCIe_40GB"
      fi
  elif [[ "$gpu_lower" == *"a100"* ]]; then
      # Generic A100, try to detect memory
      local mem_size=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
      local mem_gb=$(echo "$mem_size" | awk '{print int($1/1024)}')
      if [ "$mem_gb" -ge 75 ]; then
          echo "A100_SXM_80GB"
      else
          echo "A100_SXM_40GB"
      fi
  elif [[ "$gpu_lower" == *"b200"* ]]; then
      echo "B200_180GB"
  elif [[ "$gpu_lower" == *"h200"* ]]; then
      echo "H200_141GB"
  elif [[ "$gpu_lower" == *"h100"* ]]; then
      echo "H100_80GB"
  elif [[ "$gpu_lower" == *"v100"* ]]; then
      echo "V100_32GB"
  elif [[ "$gpu_lower" == *"a40"* ]]; then
      echo "A40_48GB"
  else
      echo "Warning: Unknown GPU model '${gpu_name}', defaulting to H100_80GB" >&2
      echo "H100_80GB"
  fi
}

# Function to extract GPU model from filename
get_gpu_from_filename() {
    local filename="$1"
    local filename_lower=$(echo "$filename" | tr '[:upper:]' '[:lower:]')
    
    for pattern in "${!GPU_MAP[@]}"; do
        if [[ "$filename_lower" == *"$pattern"* ]]; then
            echo "${GPU_MAP[$pattern]}"
            return 0
        fi
    done
    return 0
}

# Function to extract num_partitions (npus) from MLIR file
get_npus_from_mlir() {
    local mlir_file="$1"
    local npus=$(grep -o 'mhlo.num_partitions = [0-9]*' "$mlir_file" | head -1 | grep -o '[0-9]*$')
    if [ -n "$npus" ]; then
        echo "$npus"
    else
        echo "Error: Could not find mhlo.num_partitions in ${mlir_file}" >&2
        exit 1
    fi
}

get_astra()
{
  if [ -n "${ASTRA_BIN:-}" ]; then
    if [ ! -x "${ASTRA_BIN}" ]; then
      echo "ASTRA_BIN is set but is not executable: '${ASTRA_BIN}'" >&2
      exit 1
    fi
  else
    ASTRA_BIN="$(command -v AstraSim_Analytical_Congestion_Unaware 2>/dev/null || true)"
    if [ -z "${ASTRA_BIN}" ]; then
      echo "Astra-sim binary not found. Set ASTRA_BIN, e.g.:" >&2
      echo "  export ASTRA_BIN=/path/to/AstraSim_Analytical_Congestion_Unaware" >&2
      exit 1
    fi
  fi
}

run_roofline()
{
  local mlir_file="$1"
  local node_config="$2"
  local npus="$3"
  local output_dir="$4"

  run_name=$(basename "${output_dir}")

  # Check if mlir_file exists
  if [ ! -f "${mlir_file}" ]; then
      echo "MLIR file not found: ${mlir_file}" >&2
      exit 1
  fi

  # Check if node_config exists
  if [ ! -f "${node_config}" ]; then
      echo "Node config file not found: ${node_config}" >&2
      exit 1
  fi

  # Create output directory if it doesn't exist
  mkdir -p "${output_dir}"

  estimator_config="${output_dir}/${run_name}_estimator_config.json"
  chakra_output_dir="${output_dir}/estimator_out"
  estimator_run_out="${chakra_output_dir}/run.out"
  
  echo "Running HeSPaS to generate Chakra input files..." >&2
  mkdir -p "${chakra_output_dir}"
  cp "${node_config}" "${estimator_config}"
  python3 -m hespas.hespas_chakra_gen "${estimator_config}" \
    --mlir_file "${mlir_file}" \
    --output "${chakra_output_dir}" \
    --split_fn "individual_split" \
    --num_npus "${npus}" \
    --merge \
    | tee "${estimator_run_out}"
  echo "Running HeSPaS complete" >&2
}

gen_xla_conf()
{ 
  echo "
  {
    \"perf_estimator\": {
      \"method\": \"xla\",
      \"hardware\": {
        \"translate\": {
          \"target_backend\": \"cuda\"
        }
      },
      \"in_memory_only_cache\": true
    }
  }
  "
}

run_profiling()
{
  local mlir_file="$1"
  local npus="$2"
  local output_dir="$3"

  run_name=$(basename "${output_dir}")

  # Check if MLIR file exists
  if [ ! -f "${mlir_file}" ]; then
      echo "MLIR file not found: ${mlir_file}" >&2
      exit 1
  fi

  # Create output directory if it doesn't exist
  mkdir -p "${output_dir}"

  # Placeholder for profiling command
  estimator_config="${output_dir}/${run_name}_estimator_config.json"
  chakra_output_dir="${output_dir}/estimator_out"
  estimator_run_out="${chakra_output_dir}/run.out"

  gen_xla_conf > "${estimator_config}"

  echo "Running HeSPaS to generate Chakra input files..." >&2
  mkdir -p "${chakra_output_dir}"
  python3 -m hespas.hespas_chakra_gen "${estimator_config}" \
    --mlir_file "${mlir_file}" \
    --output "${chakra_output_dir}" \
    --split_fn "individual_split" \
    --num_npus "${npus}" \
    --merge \
    > "${estimator_run_out}"
  echo "Running HeSPaS complete" >&2

} 

run_astra()
{
  get_astra

  local chakra_dir="$1"
  local system_config="$2"
  local npus="$3"
  local output_dir="$4"

  run_name=$(basename "${output_dir}")
  config_name=$(basename "${system_config}")

  astra_sim_dir="${output_dir}/astrasim"
  astra_sim_out="${astra_sim_dir}/run.out"
  comm_group_config="${chakra_dir}/comm_group.json"
  echo "Running Astra-sim congestion-unaware to generate performance estimations..." >&2
  mkdir -p "${astra_sim_dir}"
  cp "${system_config}/astra-sim"/* "${astra_sim_dir}"
  astra_comm_group_arg=""
  if [ -s "${comm_group_config}" ]; then
      astra_comm_group_arg="--comm-group-configuration=${comm_group_config}"
  fi
  (
    cd "${astra_sim_dir}"
    "${ASTRA_BIN}" ${astra_comm_group_arg} \
        --workload-configuration="${chakra_dir}/dev" \
        --system-configuration="${astra_sim_dir}/system.json" \
        --remote-memory-configuration="${astra_sim_dir}/remote_memory.json" \
        --network-configuration="${astra_sim_dir}/network.yml" \
        > "${astra_sim_out}"
  )
  echo "Running Astra-sim complete" >&2

  echo "Collecting Astra-sim and HeSPaS statistics" >&2
  RUNTIME="$(sed -n "s/.*sys\[0\]\s*finished\s*,\s*\([0-9]\+\).*/\1/p" < "${astra_sim_out}")"
  if [ -z "${RUNTIME}" ]; then
      echo "Warning: Could not parse RUNTIME from astra-sim output" >&2
      echo "Check ${astra_sim_out} for output format" >&2
      return 0
  fi
  RUNTIME_S="$(echo "scale=3; ${RUNTIME} / 1000000000" | bc)"
  EXPOSED_COMMS_TIME="$(sed -n "s/.*sys\[0\]\s*finished\s*,\s[0-9]\+\s*cycles\s*,\s*exposed\s*communication\s*\([0-9]\+\).*/\1/p" < "${astra_sim_out}")"
  if [ -z "${EXPOSED_COMMS_TIME}" ]; then
      echo "Warning: Could not parse EXPOSED_COMMS_TIME from astra-sim output" >&2
      EXPOSED_COMMS_TIME="0"
  fi
  EXPOSED_COMMS_PERCENT="$(echo "scale=3; ${EXPOSED_COMMS_TIME} / ${RUNTIME}" | bc)"
  
  # These variables may not be set if run_astra is called standalone
  estimator_run_out="${output_dir}/estimator_out/run.out"
  if [ -f "${estimator_run_out}" ]; then
      FLOPS_UTIL="$(grep -i "flops/s utilisation:" "${estimator_run_out}" | cut -d ':' -f 2 | tr -s '[:space:]' | tr -d ' ' || true)"
      MEM_BW_UTIL="$(grep -i "memory bandwidth utilisation:" "${estimator_run_out}" | cut -d ':' -f 2 | tr -s '[:space:]' | tr -d ' ' || true)"
  else
      FLOPS_UTIL=""
      MEM_BW_UTIL=""
  fi
  
  if [ -n "${START_TIME:-}" ]; then
      TIME_TAKEN="$(echo "$(date '+%s')-${START_TIME}" | bc)"
  else
      TIME_TAKEN=""
  fi
  echo "${config_name},${run_name},${RUNTIME},${RUNTIME_S},${EXPOSED_COMMS_TIME},${EXPOSED_COMMS_PERCENT},${FLOPS_UTIL},${MEM_BW_UTIL},${TIME_TAKEN}"
  echo "${run_name} complete" >&2
}
