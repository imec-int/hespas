# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import pytest
import os
if __name__ == "__main__":
    import sys
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    sys.path.insert(0, repo_path)
from src.hespas.mlir_parser import MLIRParser
from src.hespas.estimator.choice import get_estimator

ZIGZAG_FIXTURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "fixtures", "zigzag"))

mlir_qkv = """
module {
  // 1) Fused QKV projection: [B,T,D] x [D,3D] -> [B,T,3D]
  func.func public @main(%x: tensor<2x1024x2048xbf16>, %w_qkv: tensor<2048x6144xbf16>) -> tensor<2x1024x6144xbf16> {
    %0 = "stablehlo.dot_general"(%x, %w_qkv) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [],
        rhs_batching_dimensions = [],
        lhs_contracting_dimensions = [2],
        rhs_contracting_dimensions = [0]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<2x1024x2048xbf16>, tensor<2048x6144xbf16>) -> tensor<2x1024x6144xbf16>
    return %0 : tensor<2x1024x6144xbf16>
  }
}

"""

mlir_attn_scores = """
module {
  // 2) Attention scores Q @ K^T: [B,H,T,hd] x [B,H,hd,T] -> [B,H,T,T]
  func.func public @main(%q: tensor<2x16x1024x128xbf16>, %kT: tensor<2x16x128x1024xbf16>) -> tensor<2x16x1024x1024xbf16> {
    %0 = "stablehlo.dot_general"(%q, %kT) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [0, 1],
        rhs_batching_dimensions = [0, 1],
        lhs_contracting_dimensions = [3],
        rhs_contracting_dimensions = [2]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<2x16x1024x128xbf16>, tensor<2x16x128x1024xbf16>) -> tensor<2x16x1024x1024xbf16>
    return %0 : tensor<2x16x1024x1024xbf16>
  }
}

"""

mlir_attn_context = """
module {
  // 3) Context = probs @ V: [B,H,T,T] x [B,H,T,hd] -> [B,H,T,hd]
  func.func public @main(%p: tensor<2x16x1024x1024xbf16>, %v: tensor<2x16x1024x128xbf16>) -> tensor<2x16x1024x128xbf16> {
    %0 = "stablehlo.dot_general"(%p, %v) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [0, 1],
        rhs_batching_dimensions = [0, 1],
        lhs_contracting_dimensions = [3],
        rhs_contracting_dimensions = [2]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<2x16x1024x1024xbf16>, tensor<2x16x1024x128xbf16>) -> tensor<2x16x1024x128xbf16>
    return %0 : tensor<2x16x1024x128xbf16>
  }
}
"""

mlir_ffn_up = """
module {
  // 4) FFN up-projection: [B,T,D] x [D,4D] -> [B,T,4D]
  func.func public @main(%x: tensor<2x1024x2048xbf16>, %w_up: tensor<2048x8192xbf16>) -> tensor<2x1024x8192xbf16> {
    %0 = "stablehlo.dot_general"(%x, %w_up) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [],
        rhs_batching_dimensions = [],
        lhs_contracting_dimensions = [2],
        rhs_contracting_dimensions = [0]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<2x1024x2048xbf16>, tensor<2048x8192xbf16>) -> tensor<2x1024x8192xbf16>
    return %0 : tensor<2x1024x8192xbf16>
  }
}
"""

mlir_out_proj = """
module {
  // 5) Output projection: [B,T,D] x [D,D] -> [B,T,D]
  func.func public @main(%y: tensor<2x1024x2048xbf16>, %w_o: tensor<2048x2048xbf16>) -> tensor<2x1024x2048xbf16> {
    %0 = "stablehlo.dot_general"(%y, %w_o) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [],
        rhs_batching_dimensions = [],
        lhs_contracting_dimensions = [2],
        rhs_contracting_dimensions = [0]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<2x1024x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<2x1024x2048xbf16>
    return %0 : tensor<2x1024x2048xbf16>
  }
}
"""

mlir_matmul = """
module {
  // Simple matmul: [M,K] x [K,N] -> [M,N]
  func.func public @main(%A: tensor<64x512xbf16>, %B: tensor<512x64xbf16>) -> tensor<64x64xbf16> {
    %0 = "stablehlo.dot_general"(%A, %B) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [],
        rhs_batching_dimensions = [],
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<64x512xbf16>, tensor<512x64xbf16>) -> tensor<64x64xbf16>
    return %0 : tensor<64x64xbf16>
  }
}
"""


mlir_dot_general = {"mlir_matmul": mlir_matmul, "mlir_qkv": mlir_qkv, "mlir_attn_scores": mlir_attn_scores, "mlir_attn_context": mlir_attn_context, "mlir_ffn_up": mlir_ffn_up, "mlir_out_proj": mlir_out_proj}

@pytest.mark.parametrize("mlir", list(mlir_dot_general.keys()))
def test_roofline_tpu_like(mlir, from_main=False):
    # tpu v3
    roofline_estimator = get_estimator("roofline")(
        {
            "peak_flops": 12e13,
            "memory_bandwidth": 9e11,
            "in_memory_only_cache": True
        }
    )
    mlir_parser = MLIRParser(mlir_string = mlir_dot_general[mlir])
    mlir_module_ops = mlir_parser.ops_list
    time = 0.0
    for op in mlir_module_ops:
        time += roofline_estimator._Estimator__get_op_estimate(op).runtime_estimate
        if from_main:
            print(op.op_name, time)
    assert time > 0
    if from_main:
        print(f"Roofline TPU v3 time (us): {time*1e6}")
    if from_main:
        return time


@pytest.mark.zigzag_onnx
@pytest.mark.parametrize("mlir", list(mlir_dot_general.keys()))
def test_zigzag_onnx_tpu_like(mlir, from_main=False):
    accelerator = os.path.join(ZIGZAG_FIXTURES_DIR, "accelerator", "tpuv3_like.yaml")
    mapping = os.path.join(ZIGZAG_FIXTURES_DIR, "mapping", "tpuv3_like.yaml")
    frequency = 9.4e8 * 2
    zigzag_onnx_estimator = get_estimator("zigzag_onnx")({
        "roofline": {
            "peak_flops": 12e13,
            "memory_bandwidth": 9e11
        },
        "accelerator": accelerator,
        "mapping": mapping,
        "frequency": frequency,
        "in_memory_only_cache": True
    })

    mlir_parser = MLIRParser(mlir_string = mlir_dot_general[mlir])
    mlir_module_ops = mlir_parser.ops_list
    time = 0.0
    for op in mlir_module_ops:
        time += zigzag_onnx_estimator._Estimator__get_op_estimate(op).runtime_estimate
    assert time > 0
    if from_main:
        print(f"ZigZag ONNX time (us): {time}")
    if from_main:
        return time


@pytest.mark.cocossim
@pytest.mark.parametrize("mlir", list(mlir_dot_general.keys()))
def test_cocossim_tpu_like(mlir, from_main=False):
    frequency = 1
    cocossim_estimator = get_estimator("cocossim")({
        "roofline": {
            "peak_flops": 12e13,
            "memory_bandwidth": 9e11
        },
        "frequency": frequency,
        "in_memory_only_cache": True
    })

    mlir_parser = MLIRParser(mlir_string = mlir_dot_general[mlir])
    mlir_module_ops = mlir_parser.ops_list
    time = 0.0
    for op in mlir_module_ops:
        if from_main:
            print(op)
        time += cocossim_estimator._Estimator__get_op_estimate(op).runtime_estimate
    assert time > 0
    if from_main:
        print(f"CocosSim time (us): {time}")
    if from_main:
        return time

if __name__ == "__main__":
    # collect times into table
    times = {"Roofline": [], "ZigZag ONNX": [], "Cocossim": []}
    for mlir in mlir_dot_general:
        times["Roofline"].append(test_roofline_tpu_like(mlir, from_main=True))
        times["ZigZag ONNX"].append(test_zigzag_onnx_tpu_like(mlir, from_main=True))
        times["Cocossim"].append(test_cocossim_tpu_like(mlir, from_main=True))
    print("\nOperation\tRoofline (us)\tZigZag ONNX (us)\tCocossim (us)")

    for i in range(len(mlir_dot_general)):
        print(f"{i+1}\t{times['Roofline'][i]}\t{times['ZigZag ONNX'][i]}\t{times['Cocossim'][i]}")
