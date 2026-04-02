# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT

import sys
import os
import json
import time
from hespas.mlir_parser.mlir_module import MLIRModule
from hespas.estimator.choice import get_estimator

"""Generate MLIR matmul examples for different memory footprints."""
# bf16 = 2 bytes per element
# Total memory ≈ M*K + K*N + M*N elements * 2 bytes
# For square-ish matrices: M≈N, so ≈ 2*M*K + M² elements
matmul_examples = {
    64: {"M": 64, "K": 64, "N": 64,
            "memory_kb": ((64*64) + (64*64) + (64*64)) * 2 / 1024,  # ≈ 0.75 KB
        },
    128: {"M": 128, "K": 128, "N": 128,
            "memory_kb": ((128*128) + (128*128) + (128*128)) * 2 / 1024,  # ≈ 3.0 KB
        },
    256: {"M": 256, "K": 256, "N": 256,
            "memory_kb": ((256*256) + (256*256) + (256*256)) * 2 / 1024,  # ≈ 12.0 KB
        },
    512: {"M": 512, "K": 512, "N": 512,
            "memory_kb": ((512*512) + (512*512) + (512*512)) * 2 / 1024,  # ≈ 48.0 KB
        },
    1024: {"M": 1024, "K": 1024, "N": 1024,
            "memory_kb": ((1024*1024) + (1024*1024) + (1024*1024)) * 2 / 1024,  # ≈ 192.0 KB
        },
    2048: {"M": 2048, "K": 2048, "N": 2048,
            "memory_kb": ((2048*2048) + (2048*2048) + (2048*2048)) * 2 / 1024,  # ≈ 768.0 KB
        },
    4096: {"M": 4096, "K": 4096, "N": 4096,
            "memory_kb": ((4096*4096) + (4096*4096) + (4096*4096)) * 2 / 1024,  # ≈ 3072.0 KB
        }
}

def generate_mlir_matmul(M=64, K=512, N=64, dtype="bf16"):
    """Generate MLIR matmul with parameterizable tensor dimensions.
    Args:
        M: Number of rows in A and output (default: 64)
        K: Number of columns in A, rows in B (default: 512) 
        N: Number of columns in B and output (default: 64)
        dtype: Data type (default: "bf16")
    Returns:
        str: MLIR string with specified tensor dimensions
    """
    return f"""module {{
    func.func public @main(%A: tensor<{M}x{K}x{dtype}>, %B: tensor<{K}x{N}x{dtype}>) -> tensor<{M}x{N}x{dtype}> {{
     %0 = "stablehlo.dot_general"(%A, %B) {{
      dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [],
        rhs_batching_dimensions = [],
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]
      >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    }} : (tensor<{M}x{K}x{dtype}>, tensor<{K}x{N}x{dtype}>) -> tensor<{M}x{N}x{dtype}>
    return %0 : tensor<{M}x{N}x{dtype}>
    }} }}"""

def sweep_get_estimator(name, config_path):
    hw_config = {}
    if config_path is not None:
        with open(config_path) as f:
            config = json.load(f)
        hw_config = config["perf_estimator"]['hardware']
    return get_estimator(name)(hw_config)

def main():
    text_file_out = os.path.join(os.path.dirname(__file__), "systolic_experiment_results_debug.txt")
    json_out = os.path.join(os.path.dirname(__file__), "systolic_experiment_results_debug.json")
    config_paths = {"onnxim": os.path.join(os.path.dirname(__file__), "config_onnxim_tpuv3.json")}
    estimators = {k: sweep_get_estimator(k, v) for k, v in config_paths.items()}
    result = {}
    strings = {}

    for size_name, config in matmul_examples.items():
        result[size_name] = {
            "memory_kb": config["memory_kb"]
        }
        strings[size_name] = {}
        mlir_matmul = generate_mlir_matmul(config["M"], config["K"], config["N"])
        module = MLIRModule(mlir_string=mlir_matmul)
        for estimator_name, estimator in estimators.items():
            wall_time_start = time.perf_counter()
            estimate = estimator.get_estimate(module)
            wall_time = time.perf_counter() - wall_time_start
            runtime = estimate.runtime_estimate if estimate.runtime_estimate is not None else 0.0
            result[size_name]["{}_time_s".format(estimator_name)] = runtime
            strings[size_name]["{} latency".format(estimator_name)] = "{}s".format(runtime)
            result[size_name]["{}_total_time_s".format(estimator_name)] = wall_time
            strings[size_name]["{} simulation time".format(estimator_name)] = f"{wall_time:.6f} s"

        with open(text_file_out, "a") as f:
            f.write(f"\n--- Results for {size_name} at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            f.write(f"Size: {size_name} \n")
            for estimator_string, result_string in strings[size_name].items():
                f.write(f"{estimator_string}: {result_string}\n")
            f.write("----------------------------------------\n")

    with open(json_out, "w") as f:
        json.dump(result, f, indent=4)
    print(result)

if __name__ == "__main__":
    main()
