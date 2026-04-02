# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import math
import pytest
from src.hespas.estimator.roofline_estimator import RooflineEstimator, RooflineMissingDatatypeError
from src.hespas.utils.dt_sizes import get_aligned_bytes
from src.hespas.utils.op_info import OpInfo
from src.hespas.mlir_parser.mlir_module import MLIRModule

def test_op_info_bytes_conversion_1():
    # Test OpInfo byte calculations for different data types
    op_info = OpInfo(
        op_name='',
        input_types=[((10, 10), 'f32'), ((5, 10), 'f32')],
        output_types=[((1, 10), 'f32')]
    )
    assert op_info.get_input_bytes(0) == math.prod((10, 10)) * 4  # f32 = 4 bytes
    assert op_info.get_input_bytes(1) == math.prod((5, 10)) * 4
    assert op_info.get_output_bytes(0) == math.prod((1, 10)) * 4


def test_op_info_bytes_conversion_2():
    # Test OpInfo byte calculations for different data types
    op_info = OpInfo(
        op_name='',
        input_types=[((10, 10, 10), 'f16'), ((5, 10), 'i64')],
        output_types=[((1, 10), 'bf16')]
    )
    assert op_info.get_input_bytes(0) == math.prod((10, 10, 10)) * 2  # f16 = 2 bytes
    assert op_info.get_input_bytes(1) == math.prod((5, 10)) * 8  # i64 = 8 bytes
    assert op_info.get_output_bytes(0) == math.prod((1, 10)) * 2


# Test get_op_run_time_estimate for an elementwise binary op
def test_roofline_elementwise_runtime():
    hw_config = {'peak_flops': 100.0, 'memory_bandwidth': 50.0, 'in_memory_only_cache': True}
    estimator = RooflineEstimator(hw_config)
    # op: add, shape (10, 10), dtype f32
    op_info = OpInfo(
        op_name='stablehlo.add',
        input_types=[((10, 10), 'f32'), ((10, 10), 'f32')],
        output_types=[((10, 10), 'f32')]
    )
    runtime = estimator._Estimator__get_op_estimate(op_info).runtime_estimate
    # flops = 100, bytes = 3x400=1200, compute_time=100/100, mem_time=1200/50, so runtime=24
    assert runtime == 24


# Test get_op_run_time_estimate for a free op (constant)
def test_roofline_free_op_runtime():
    hw_config = {'peak_flops': 100.0, 'memory_bandwidth': 50.0, 'in_memory_only_cache': True}
    estimator = RooflineEstimator(hw_config)
    op_info = OpInfo(
        op_name='stablehlo.constant',
        input_types=[],
        output_types=[((1,), 'f32')]
    )
    runtime = estimator._Estimator__get_op_estimate(op_info).runtime_estimate
    assert runtime == 0.0


def test_roofline_reduce():
    hw_config = {'peak_flops': 100.0, 'memory_bandwidth': 5.0, 'in_memory_only_cache': True}
    estimator = RooflineEstimator(hw_config)
    op_info = OpInfo(
        op_name='stablehlo.reduce',
        input_types=[((2, 3, 4 ), 'f32')],
        output_types=[((2, 3), 'f32')]
    )
    op_info.dimensions = [2]

    flops = (4 - 1) * 2 * 3
    bytes = 4 * 2 * 3 * 4 + 4 * 2 * 3
    res = max(flops / hw_config['peak_flops'], bytes / hw_config['memory_bandwidth'])
    runtime = estimator._Estimator__get_op_estimate(op_info).runtime_estimate
    # Assuming a simple reduction, flops = 10, bytes = 400 + 40 = 440
    assert runtime == pytest.approx(res)  # Adjusted for expected runtime


def test_roofline_noflop_ops():
    hw_config = {'peak_flops': 100.0, 'memory_bandwidth': 2.0, 'in_memory_only_cache': True}
    estimator = RooflineEstimator(hw_config)
    op_info = OpInfo(
        op_name='stablehlo.transpose',
        input_types=[((4,3,2,1), 'f16')],
        output_types=[((1,2,3,4), 'f16')]
    )
    runtime = estimator._Estimator__get_op_estimate(op_info).runtime_estimate
    bytes = 4 * 3 * 2 * 1 * 2 + 1 * 2 * 3 * 4 * 2
    assert runtime == bytes / hw_config['memory_bandwidth']


def test_roofline_unary_ops():
    hw_config = {'peak_flops': 100.0, 'memory_bandwidth': 50.0, 'in_memory_only_cache': True}
    estimator = RooflineEstimator(hw_config)
    op_info = OpInfo(
        op_name='stablehlo.negate',
        input_types=[((1, 2, 3), 'f32')],
        output_types=[((1, 2, 3), 'f32')]
    )
    runtime = estimator._Estimator__get_op_estimate(op_info).runtime_estimate
    flops = math.prod((1, 2, 3))  # Assuming 1 flop per element for unary ops
    bytes = 1 * 2 * 3 * 4 + 1 * 2 * 3 * 4  # input + output
    exp = max(flops / hw_config['peak_flops'], bytes / hw_config['memory_bandwidth'])
    assert runtime == pytest.approx(exp)


def test_roofline_select():
    hw_config = {'peak_flops': 100.0, 'memory_bandwidth': 50.0, 'in_memory_only_cache': True}
    estimator = RooflineEstimator(hw_config)
    op_info = OpInfo(
        op_name='stablehlo.select',
        input_types=[((10, 10), 'f32'), ((10, 10), 'f32'), ((10, 10), 'f32')],
        output_types=[((10, 10), 'f32')]
    )
    runtime = estimator._Estimator__get_op_estimate(op_info).runtime_estimate
    flops = math.prod((10, 10))  # Assuming 1 flop per element for select
    bytes = (3 * 10 * 10 * 4) + (10 * 10 * 4)  # input + output
    exp = max(flops / hw_config['peak_flops'], bytes / hw_config['memory_bandwidth'])
    assert runtime == pytest.approx(exp)


def test_roofline_dot_general_1():
    hw_config = {'peak_flops': 100.0, 'memory_bandwidth': 50.0, 'in_memory_only_cache': True}
    estimator = RooflineEstimator(hw_config)
    op_info = OpInfo(
        op_name='stablehlo.dot_general',
        input_types=[((10, 5), 'f32'), ((5, 10), 'f32')],
        output_types=[((10, 10), 'f32')]
    )
    op_info.lhs_dims = {0: 10, 1: 5}
    op_info.rhs_dims = {0: 5, 1: 10}
    op_info.dims = {'lhs': [1], 'rhs': [0], 'batch': []}

    runtime = estimator._Estimator__get_op_estimate(op_info).runtime_estimate
    flops = 2 * (10 * 10) * (5)  # FLOPs for dot_general
    bytes = (4 * 10 * 5) + (4 * 5 * 10) + (4 * 10 * 10)  # input + input + output
    exp = max(flops / hw_config['peak_flops'], bytes / hw_config['memory_bandwidth'])
    assert runtime == pytest.approx(exp)


def test_roofline_dot_general_2():  # with batch
    hw_config = {'peak_flops': 100.0, 'memory_bandwidth': 50.0, 'in_memory_only_cache': True}
    estimator = RooflineEstimator(hw_config)
    op_info = OpInfo(
        op_name='stablehlo.dot_general',
        input_types=[((10, 6, 5), 'f32'), ((10, 5, 10), 'f32')],
        output_types=[((10, 6, 10), 'f32')]
    )
    op_info.lhs_dims = {0: 10, 1: 6, 2: 5}
    op_info.rhs_dims = {0: 10, 1: 5, 2: 10}
    op_info.dims = {'lhs': [2], 'rhs': [1], 'batch': [(0, 0)]}

    # FLOPs: 2 * output_elements * contracting_size
    flops = 2 * (10 * 6 * 10) * 5  # B × M × N × K
    # Bytes: 4 bytes per float
    bytes = (
        4 * 10 * 6 * 5 +        # LHS: 10×5
        4 * 10 * 5 * 10 +   # RHS: 10×5×10
        4 * 10 * 6 * 10    # Output: 10×10×10
    )
    exp = max(flops / hw_config['peak_flops'], bytes / hw_config['memory_bandwidth'])
    runtime = estimator._Estimator__get_op_estimate(op_info).runtime_estimate
    assert runtime == pytest.approx(exp)


def test_roofline_concatenate():
    hw_config = {'peak_flops': 100.0, 'memory_bandwidth': 50.0, 'in_memory_only_cache': True}
    estimator = RooflineEstimator(hw_config)
    op_info = OpInfo(
        op_name='stablehlo.concatenate',
        input_types=[((1, 3), 'f32'), ((2, 3), 'f32')],
        output_types=[((3, 3), 'f32')]
    )
    runtime = estimator._Estimator__get_op_estimate(op_info).runtime_estimate
    # Concatenate is a zero-FLOP, memory-bound op
    input_bytes = (1 * 3 * 4) + (2 * 3 * 4)  # two inputs
    output_bytes = (3 * 3 * 4)  # one output
    total_bytes = input_bytes + output_bytes
    assert runtime == total_bytes / hw_config['memory_bandwidth']


def test_roofline_mlir_convolution():
    mlir_content = """func.func @main(%lhs: tensor<1x4x4x1xi64>, %rhs: tensor<3x3x1x1xi64>) -> (tensor<1x2x2x1xi64>) {

  %result = stablehlo.convolution(%lhs, %rhs)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [4, 4],
      lhs_dilate = [2, 2]
    } {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    }
  : (tensor<1x4x4x1xi64>, tensor<3x3x1x1xi64>) -> tensor<1x2x2x1xi64>
  return %result : tensor<1x2x2x1xi64>
}
"""
    def roofline_mlir_convolution(hw_config):
        mlir_module = MLIRModule(mlir_string=mlir_content, parent_module="inline")
        estimator = RooflineEstimator(hw_config)

        obs_runtime = estimator.get_estimate(mlir_module).runtime_estimate

        kernel_mult = 2 * 3 * 3
        exp_flops = kernel_mult * 2 * 2
        exp_bytes = (4*4 + 3*3 + 2*2 ) * 8
        exp_runtime = max(exp_flops/hw_config["peak_flops"], exp_bytes/hw_config["memory_bandwidth"])

        print("RUNTIME:", "OBS:", obs_runtime, "EXP:", exp_runtime)
        assert obs_runtime == exp_runtime

    # compute bound
    roofline_mlir_convolution({'peak_flops': 1.0, 'memory_bandwidth': 1000.0, 'in_memory_only_cache': True})
    #mem bound
    roofline_mlir_convolution({'peak_flops': 1000.0, 'memory_bandwidth': 1.0, 'in_memory_only_cache': True})


def test_roofline_mlir_abs():
    mlir_content = """func.func @main(%input : tensor<2xf32>) -> (tensor<2xf32>) {
  %result = stablehlo.abs %input : tensor<2xf32>
  return %result : tensor<2xf32>
}
"""
    mlir_module = MLIRModule(mlir_string=mlir_content, parent_module="inline")

    hw_config = {'peak_flops': 100.0, 'memory_bandwidth': 50.0, 'in_memory_only_cache': True}
    estimator = RooflineEstimator(hw_config)

    obs_runtime = estimator.get_estimate(mlir_module).runtime_estimate

    exp_flops = 2
    exp_bytes = 2 * 2 * 4
    exp_runtime = max(exp_flops/hw_config["peak_flops"], exp_bytes/hw_config["memory_bandwidth"])

    print("RUNTIME:", "OBS:", obs_runtime, "EXP:", exp_runtime)
    assert obs_runtime == exp_runtime


def test_roofline_mlir_dot_general():
    mlir_content = """module {
  func.func @main(%lhs: tensor<2x2x2xi64>, %rhs: tensor<2x2x2xi64>) -> tensor<2x2x2xi64> {

    %result = stablehlo.dot_general %lhs, %rhs,
        batching_dims = [0] x [0],
        contracting_dims = [2] x [1],
        precision = [DEFAULT, DEFAULT]
        : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>

    return %result : tensor<2x2x2xi64>
  }
}
"""
    mlir_module = MLIRModule(mlir_string=mlir_content, parent_module="inline")

    hw_config = {'peak_flops': 100.0, 'memory_bandwidth': 50.0, 'in_memory_only_cache': True}
    estimator = RooflineEstimator(hw_config)

    obs_runtime = estimator.get_estimate(mlir_module).runtime_estimate

    exp_flops = 2 * 2 ** 4 # 2 * BNMK
    exp_bytes = (2**3 + 2**3 + 2**3) * 8
    exp_runtime = max(exp_flops/hw_config["peak_flops"], exp_bytes/hw_config["memory_bandwidth"])

    print("RUNTIME:", "OBS:", obs_runtime, "EXP:", exp_runtime)
    assert obs_runtime == exp_runtime

def test_roofline_mlir_dot_general_multiprec_basic():
    mlir_content_i64 = """module {
  func.func @main(%lhs: tensor<2x2x2xi64>, %rhs: tensor<2x2x2xi64>) -> tensor<2x2x2xi64> {

    %result = stablehlo.dot_general %lhs, %rhs,
        batching_dims = [0] x [0],
        contracting_dims = [2] x [1],
        precision = [DEFAULT, DEFAULT]
        : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>

    return %result : tensor<2x2x2xi64>
  }
}
"""

    mlir_content_bf16 = """module {
  func.func @main(%lhs: tensor<2x2x2xbf16>, %rhs: tensor<2x2x2xbf16>) -> tensor<2x2x2xbf16> {

    %result = stablehlo.dot_general %lhs, %rhs,
        batching_dims = [0] x [0],
        contracting_dims = [2] x [1],
        precision = [DEFAULT, DEFAULT]
        : (tensor<2x2x2xbf16>, tensor<2x2x2xbf16>) -> tensor<2x2x2xbf16>

    return %result : tensor<2x2x2xbf16>
  }
}
"""

    mlir_module_i64 = MLIRModule(mlir_string=mlir_content_i64, parent_module="inline")
    mlir_module_bf16 = MLIRModule(mlir_string=mlir_content_bf16, parent_module="inline")

    hw_config = {
        'peak_flops': 100.0,
        'per_datatype_flops': {
            'bf16': 200.0
        },
        'memory_bandwidth': 50.0,
        'in_memory_only_cache': True
    }
    estimator = RooflineEstimator(hw_config)

    obs_runtime_i64 = estimator.get_estimate(mlir_module_i64).runtime_estimate
    obs_runtime_bf16 = estimator.get_estimate(mlir_module_bf16).runtime_estimate

    exp_flops = 2 * 2 ** 4 # 2 * BNMK
    exp_bytes_i64 = (2**3 + 2**3 + 2**3) * 8
    exp_bytes_bf16 = (2**3 + 2**3 + 2**3) * 2

    exp_runtime_i64 = max(exp_flops/hw_config["peak_flops"], exp_bytes_i64/hw_config["memory_bandwidth"])
    exp_runtime_bf16 = max(exp_flops/hw_config["per_datatype_flops"]["bf16"], exp_bytes_bf16/hw_config["memory_bandwidth"])

    print("RUNTIME i64:", "OBS:", obs_runtime_i64, "EXP:", exp_runtime_i64)
    print("RUNTIME bf16:", "OBS:", obs_runtime_bf16, "EXP:", exp_runtime_bf16)
    assert obs_runtime_i64 == exp_runtime_i64
    assert obs_runtime_bf16 == exp_runtime_bf16

def test_roofline_mlir_dot_general_multiprec_error():
    mlir_content_i64 = """module {
  func.func @main(%lhs: tensor<2x2x2xi64>, %rhs: tensor<2x2x2xi64>) -> tensor<2x2x2xi64> {

    %result = stablehlo.dot_general %lhs, %rhs,
        batching_dims = [0] x [0],
        contracting_dims = [2] x [1],
        precision = [DEFAULT, DEFAULT]
        : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>

    return %result : tensor<2x2x2xi64>
  }
}
"""

    mlir_content_bf16 = """module {
  func.func @main(%lhs: tensor<2x2x2xbf16>, %rhs: tensor<2x2x2xbf16>) -> tensor<2x2x2xbf16> {

    %result = stablehlo.dot_general %lhs, %rhs,
        batching_dims = [0] x [0],
        contracting_dims = [2] x [1],
        precision = [DEFAULT, DEFAULT]
        : (tensor<2x2x2xbf16>, tensor<2x2x2xbf16>) -> tensor<2x2x2xbf16>

    return %result : tensor<2x2x2xbf16>
  }
}
"""

    mlir_module_i64 = MLIRModule(mlir_string=mlir_content_i64, parent_module="inline")
    mlir_module_bf16 = MLIRModule(mlir_string=mlir_content_bf16, parent_module="inline")

    hw_config = {
        'peak_flops': 100.0,
        'per_datatype_flops': {
            'bf16': 200.0
        },
        'memory_bandwidth': 50.0,
        'in_memory_only_cache': True,
        'error_on_unknown_type': True
    }
    estimator = RooflineEstimator(hw_config)

    obs_runtime_bf16 = estimator.get_estimate(mlir_module_bf16).runtime_estimate

    exp_flops = 2 * 2 ** 4 # 2 * BNMK
    exp_bytes_bf16 = (2**3 + 2**3 + 2**3) * 2

    exp_runtime_bf16 = max(exp_flops/hw_config["per_datatype_flops"]["bf16"], exp_bytes_bf16/hw_config["memory_bandwidth"])
    print("RUNTIME bf16:", "OBS:", obs_runtime_bf16, "EXP:", exp_runtime_bf16)
    assert obs_runtime_bf16 == exp_runtime_bf16

    with pytest.raises(RooflineMissingDatatypeError):
        estimator.get_estimate(mlir_module_i64).runtime_estimate

def test_roofline_mlir_dot_general_multiprec():
    tensor_type_tmp = "tensor<2x2x2x{}>"
    input_args_tmp = "%lhs_{0}: tensor<2x2x2x{0}>, %rhs_{0}: tensor<2x2x2x{0}>"
    op_tmp = """%result_{0} = stablehlo.dot_general %lhs_{0}, %rhs_{0},
        batching_dims = [0] x [0],
        contracting_dims = [2] x [1],
        precision = [DEFAULT, DEFAULT]
        : (tensor<2x2x2x{0}>, tensor<2x2x2x{0}>) -> tensor<2x2x2x{0}>"""

    return_name_tmp = "%result_{}"
    func_tmp = """module {{
      func.func @main({args}) -> ({return_types}) {{

      {ops}

    return {return_names} : {return_types}
  }}
}}
    """

    def any_join(delimit, template, dtypes):
        return delimit.join([template.format(x) for x in dtypes])

    def comma_join(template, dtypes):
        return any_join(", ", template, dtypes)

    def newline_join(template, dtypes):
        return any_join("\n", template, dtypes)

    def get_args(dtypes):
        return comma_join(input_args_tmp, dtypes)

    def get_ret_args(dtypes):
        return comma_join(tensor_type_tmp, dtypes)

    def get_ret_names(dtypes):
        return comma_join(return_name_tmp, dtypes)

    def get_ops(dtypes):
        return newline_join(op_tmp, dtypes)

    def get_mlir(dtypes):
        return func_tmp.format(args=get_args(dtypes), return_types=get_ret_args(dtypes), ops=get_ops(dtypes), return_names=get_ret_names(dtypes))

    baseline_flops = 1
    dtypes = {
        'i64': 100,
        'f32': 50,
        'f64': 10,
        'bf16': 200
    }

    mlir_module = MLIRModule(mlir_string=get_mlir(list(dtypes.keys())), parent_module="inline")

    hw_config = {
        'peak_flops': baseline_flops,
        'per_datatype_flops': dtypes,
        'memory_bandwidth': 50.0,
        'in_memory_only_cache': True,
        'error_on_unknown_type': True
    }
    estimator = RooflineEstimator(hw_config)

    obs_runtime = estimator.get_estimate(mlir_module).runtime_estimate

    def get_exp_runtime_per_dtype(dtype, hw_config):
        exp_flops = 2 * 2 ** 4 # 2 * BNMK
        exp_bytes = (2**3 + 2**3 + 2**3) * get_aligned_bytes(dtype)
        peak_flops = hw_config.get("per_datatype_flops", {}).get(dtype, hw_config["peak_flops"])
        return max(exp_flops/peak_flops, exp_bytes/hw_config["memory_bandwidth"])

    def get_exp_runtime(dtypes, hw_config):
        return sum([get_exp_runtime_per_dtype(x, hw_config) for x in dtypes.keys()])

    exp_runtime = get_exp_runtime(dtypes, hw_config)

    print("RUNTIME:", "OBS:", obs_runtime, "EXP:", exp_runtime)
    assert obs_runtime == exp_runtime
