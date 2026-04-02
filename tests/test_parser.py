# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import shutil
from pathlib import Path
import os
import pytest

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
MLIR_DIR = os.path.join(FIXTURES_DIR, "mlir")

from src.hespas.mlir_parser import MLIRParser
from src.hespas.mlir_parser.mlir_splitter import (
    linear_split,
    SeparatorPolicy,
    parse_and_split_mlir,
)
from src.hespas.mlir_parser.mlir_common import is_dot_general_op, NodeType
from src.hespas.mlir_parser.mlir_module import MLIRModule
from src.hespas.estimator.roofline_estimator import RooflineEstimator

def test_linear_split_default():
    mlir_file = os.path.join(MLIR_DIR, 'jax_example.mlir')
    mlir_parser = MLIRParser(mlir_path=mlir_file)
    ops = mlir_parser.operations
    blocks, _ = linear_split(ops)
    assert len(blocks)==8

@pytest.mark.parametrize("separator_policy, expected_blocks", [
    (SeparatorPolicy.END_BLOCK, 8 + 1),
    (SeparatorPolicy.ISOLATED, 8 + 8 + 1),
    (SeparatorPolicy.START_BLOCK, 9),
])
def test_linear_split_dot_general(separator_policy, expected_blocks):
    mlir_file = os.path.join(MLIR_DIR, 'jax_example.mlir')
    mlir_parser = MLIRParser(mlir_path=mlir_file)
    ops = mlir_parser.operations

    blocks, _ = linear_split(ops, separator=is_dot_general_op, separator_policy=separator_policy, separator_node_type=NodeType.COMP_NODE)
    assert len(blocks) == expected_blocks


def is_multiply_op(op):
    return 'stablehlo.multiply' in str(op.name)


def is_subtract_op(op):
    return 'stablehlo.subtract' in str(op.name)


@pytest.mark.parametrize("block_lim, separator, separator_policy, expected_blocks", [
    (1, is_multiply_op, SeparatorPolicy.END_BLOCK, 10),  # 9 ops + return
    (2, is_multiply_op, SeparatorPolicy.END_BLOCK, 6),
    (2, is_multiply_op, SeparatorPolicy.START_BLOCK, 5),
    (3, is_multiply_op, SeparatorPolicy.START_BLOCK, 5),
    (3, is_subtract_op, SeparatorPolicy.START_BLOCK, 4),
    (100, is_subtract_op, SeparatorPolicy.START_BLOCK, 2),
    (100, is_subtract_op, SeparatorPolicy.ISOLATED, 3),
])
def test_linear_split_block_lim(block_lim, separator, separator_policy, expected_blocks):
    mlir_string = """
        module {
        func.func public @main(%a: tensor<f32>, %b: tensor<f32>) -> tensor<f32> {
            %1 = stablehlo.multiply %b, %a : tensor<f32>
            %2 = stablehlo.add %1, %b : tensor<f32>
            %3 = stablehlo.multiply %2, %1 : tensor<f32>
            %4 = stablehlo.add %3, %1 : tensor<f32>
            %5 = stablehlo.multiply %4, %2 : tensor<f32>
            %6 = stablehlo.subtract %5, %3 : tensor<f32>
            %7 = stablehlo.multiply %6, %4 : tensor<f32>
            %8 = stablehlo.add %7, %5 : tensor<f32>
            %9 = stablehlo.multiply %8, %6 : tensor<f32>
            return %9 : tensor<f32>
        }
    }
    """
    mlir_parser = MLIRParser(mlir_string=mlir_string)
    ops = mlir_parser.operations
    blocks, _ = linear_split(ops, block_lim=block_lim, separator=separator, separator_policy=separator_policy, separator_node_type=NodeType.COMP_NODE)
    assert len(blocks) == expected_blocks


def test_parse_convolution():
    mlir_file = os.path.join(MLIR_DIR, 'convolution.mlir')

    output_path = Path('output/tests/test_parse_conv')
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    modules = parse_and_split_mlir(mlir_file, output_path)
    assert len(modules)==1
    shutil.rmtree(output_path)

mlir_str_nested_funcs = """
    module {
        // entry -> mid1 -> leaf
        // entry -> mid2 -> mid1 -> leaf
        //               -> leaf
        func.func public @main(%a: tensor<f32>, %b: tensor<f32>) -> tensor<f32> {
            %x = call @_mid1(%a, %b) : (tensor<f32>, tensor<f32>) -> tensor<f32>
            %y = call @_mid2(%a) : (tensor<f32>) -> tensor<f32>
            %out = stablehlo.add %x, %y : tensor<f32>
            return %out : tensor<f32>
        }

        func.func private @_mid1(%x: tensor<f32>, %y: tensor<f32>) -> tensor<f32> {
            %t = stablehlo.subtract %x, %y : tensor<f32>
            %r = call @_leaf(%t, %y) : (tensor<f32>, tensor<f32>) -> tensor<f32>
            return %r : tensor<f32>
        }

        func.func private @_mid2(%x: tensor<f32>) -> tensor<f32> {
            %c = call @_leaf(%x, %x) : (tensor<f32>, tensor<f32>) -> tensor<f32>
            %r = call @_mid1(%c, %x) : (tensor<f32>, tensor<f32>) -> tensor<f32>
            return %r : tensor<f32>
        }

        func.func private @_leaf(%u: tensor<f32>, %v: tensor<f32>) -> tensor<f32> {
            %s = stablehlo.multiply %u, %v : tensor<f32>
            return %s : tensor<f32>
        }
    }
    """

def test_private_function_tree():

    mlir_parser = MLIRParser(mlir_string=mlir_str_nested_funcs)
    pf = mlir_parser.private_functions
    assert len(pf) == 3
    tree, _ = mlir_parser.get_private_functions_nx_tree(pf)
    assert tree is not None
    assert len(tree.nodes) == 3
    assert len(tree.edges) == 3

    fmap = mlir_parser.get_private_functions_map(pf)
    assert len(fmap) == 3
    assert "_mid1" in fmap
    assert "_mid2" in fmap
    assert "_leaf" in fmap
    assert "main" not in fmap

    ops_mid1 = mlir_parser.get_private_functions_ops(fmap["_mid1"])
    # print("ops_mid1:", [op.op_name for op in ops_mid1])
    assert len(ops_mid1) == 2

    ops_mid2 = mlir_parser.get_private_functions_ops(fmap["_mid2"])
    # print("ops_mid2:", [op.op_name for op in ops_mid2])
    assert len(ops_mid2) == 3

    ops_leaf = mlir_parser.get_private_functions_ops(fmap["_leaf"])
    # print("ops_leaf:", [op.op_name for op in ops_leaf])
    assert len(ops_leaf) == 1


def test_private_function_tree_roofline():

    hw_config = {'peak_flops': 100, 'memory_bandwidth': 12, 'in_memory_only_cache': True}
    estimator = RooflineEstimator(hw_config)
    runtime = estimator.get_estimate(MLIRModule(mlir_string=mlir_str_nested_funcs)).runtime_estimate
    assert runtime == 6 # 6s in us
