# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import os
from jaxlib.mlir.ir import Module, Location, FunctionType, InsertionPoint
from jaxlib.mlir.dialects import func
os.environ["TEMPORARILY_DISABLE_PROTOBUF_VERSION_CHECK"] = "true" # Hack to force Protobuf to not be as picky
from chakra.schema.protobuf.et_def_pb2 import NodeType as ChakraNodeType

import logging
import re
from enum import Enum
from functools import lru_cache
from typing import Optional

log = logging.getLogger(__name__)


NodeType = Enum("NodeType", {k: v for k, v in zip(ChakraNodeType.keys(), ChakraNodeType.values())})

comm_op_names = {
      "stablehlo.collective_permute": True,
      "stablehlo.all_gather": True,
      "stablehlo.all_to_all": True,
      "stablehlo.reduce_scatter": True,
      "stablehlo.all_reduce": True,
}

op_classifiers = {}

def add_classifier(func):
    op_classifiers[func.__name__] = func
    return func

@lru_cache
def is_communication_op_str(op_name):
    return comm_op_names.get(op_name, False)

@add_classifier
def is_convolution_op(op):
    """Check if the operation is a StableHLO convolution op."""
    return 'stablehlo.convolution' in op.OPERATION_NAME

@add_classifier
def is_dot_general_op(op):
    """Check if the operation is a StableHLO dot_general op."""
    return 'stablehlo.dot_general' in op.OPERATION_NAME

@add_classifier
def is_communication_op(op):
    """Check if the operation is a StableHLO communication op."""
    return is_communication_op_str(op.OPERATION_NAME)

@add_classifier
def is_reduce_op(op):
    """Check if the operation is a function return op."""
    return 'stablehlo.reduce' in op.OPERATION_NAME

@add_classifier
def is_return_op(op):
    """Check if the operation is a function return op."""
    return 'func.return' in op.OPERATION_NAME

def get_external_inputs(block):
    """returns a dict of argument: argument types"""
    seen_ops = set()
    arg_dict = dict()
    for i, op in enumerate(block):
        for arg in op.operands:
            arg_name = arg
            if arg_name in seen_ops:
                continue
            arg_dict[arg] = arg.type
            seen_ops.add(arg_name)
        for res in op.results:
            seen_ops.add(res)
    return arg_dict


def get_external_outputs(block):
    """returns a dict of argument_name: argument types"""
    used_ops = set()
    ret_dict = dict()
    for i, op in enumerate(reversed(block)):
        for arg in op.operands:
            arg_name = arg
            if arg_name in ret_dict.keys():
                del ret_dict[arg_name]
            used_ops.add(arg_name)
        # check if multi-returns are correctly handled
        for res in op.results:
            if res not in used_ops:
                ret_dict[res] = res.type
    return ret_dict


def get_dot_general_dimensions(dot_op):
    """Extract batch and contracting dimensions from a stablehlo.dot_general op."""

    # Get the dot_dimension_numbers attribute
    dot_dims_attr = dot_op.attributes["dot_dimension_numbers"]

    # Convert to string and parse
    dot_dims_str = str(dot_dims_attr)

    # Extract lhs_contracting_dimensions, rhs_contracting_dimensions, and batch dimensions
    lhs_contracting = re.search(r"lhs_contracting_dimensions = \[([0-9, ]+)\]", dot_dims_str)
    rhs_contracting = re.search(r"rhs_contracting_dimensions = \[([0-9, ]+)\]", dot_dims_str)
    lhs_batch_dims = re.search(r"lhs_batching_dimensions = \[([0-9, ]+)\]", dot_dims_str)
    rhs_batch_dims = re.search(r"rhs_batching_dimensions = \[([0-9, ]+)\]", dot_dims_str)

    # Convert to lists of integers
    lhs_contracting_dims = list(map(int, lhs_contracting.group(1).split(","))) if lhs_contracting else []
    rhs_contracting_dims = list(map(int, rhs_contracting.group(1).split(","))) if rhs_contracting else []
    lhs_batch_dims = list(map(int, lhs_batch_dims.group(1).split(","))) if lhs_batch_dims else []
    rhs_batch_dims = list(map(int, rhs_batch_dims.group(1).split(","))) if rhs_batch_dims else []
    batch_dims = [(lb, rb) for (lb, rb) in zip(lhs_batch_dims, rhs_batch_dims, strict=True)]

    return lhs_contracting_dims, rhs_contracting_dims, batch_dims

replica_group_re = re.compile(r"replica_groups\s*=\s*dense<(\[\[.*?\]\]|\d+)>\s*:\s*tensor<(\d+)x(\d+)xi64>")
inner_match_re = re.compile(r"\[([^\[\]]+)\]")

@lru_cache
def parse_replica_groups(mlir_string: str) -> Optional[list[list[int]]]:
    """Extract replica_groups from a StableHLO collective operation's MLIR string.

    Parses the ``replica_groups`` dense attribute that appears in collective
    operations such as ``stablehlo.all_reduce``, ``stablehlo.all_gather``, etc.

    Supported formats:
        - ``dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>``  → [[0, 1, 2, 3]]
        - ``dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>`` → [[0, 1], [2, 3]]
        - ``dense<0> : tensor<1x1xi64>``                → [[0]]

    For ``stablehlo.collective_permute`` which uses ``source_target_pairs``
    instead of ``replica_groups``, returns None.

    Returns:
        A list of groups, where each group is a list of integer device ids,
        or None if no ``replica_groups`` attribute is found.
    """
    # collective_permute uses source_target_pairs, not replica_groups
    if "stablehlo.collective_permute" in mlir_string:
        return None

    match = replica_group_re.search(mlir_string)
    if not match:
        return None
    return parse_replica_from_match(match.group(1), match.group(2), match.group(3))

@lru_cache
def parse_replica_from_match(raw_value, num_groups, group_size):
    if raw_value.startswith("[["):
        # Parse nested list: [[0, 1], [2, 3]]
        groups = []
        for inner_match in inner_match_re.finditer(raw_value):
            group = [int(x.strip()) for x in inner_match.group(1).split(",")]
            groups.append(group)
        return groups

    num_groups = int(num_groups)
    group_size = int(group_size)

    # Scalar value like dense<0> : tensor<1x1xi64>
    scalar = int(raw_value)
    if num_groups == 1 and group_size == 1:
        return [[scalar]]
    # Flatten scalar across the tensor shape (unusual but handle gracefully)
    return [[scalar] * group_size for _ in range(num_groups)]

def store_private_functions(priv_funcs, output_path, context=None):
    """
    Write private functions to a single file for debugging purposes.

    This function writes all private functions extracted by the MLIR parser
    to a file named `private_functions.mlir` in the specified output path.
    This is useful for debugging and analyzing the private functions in isolation.

    Args:
        mlir_parser: An instance of the MLIR parser that provides access to private functions.
        output_path (str): The directory path where the `private_functions.mlir` file will be saved.

    Returns:
        list: A list of private functions extracted by the MLIR parser.

    Note:
        This function must be executed within an MLIR context.
    """
    # Store private functions
    log.info("Saving private functions to: {}".format(os.path.join(output_path, "private_functions.mlir")))
    funcs_mod = append_private_functions(priv_funcs, None, context=context)
    with open(os.path.join(output_path, "private_functions.mlir"), "w") as f:
        f.write(str(funcs_mod))
    return priv_funcs


def append_private_functions(private_funcs, module=None, context=None):
    # Create a new module if not provided
    if module is None:
        module = Module.create(loc=Location.unknown(context))
    added_funcs = []
    with InsertionPoint(module.body):
        for p_func in private_funcs:
            if p_func.sym_name not in added_funcs:
                added_funcs.append(p_func.sym_name)
                module.body.append(p_func.clone())
    return module


def create_new_module_with_operations(block, private_funcs=None, context=None):
    # return op is explicitly created in the new module, so omitting the original one
    if "return" in block[-1].name:
        block = block[:-1]

    if len(block) == 0:
        log.warning("Ignoring block with a single return statement.")
        return None, None, None

    block_arg_types = get_external_inputs(block)
    ret_arg_types = get_external_outputs(block)
    # create the module and function
    # the order should matter! -> check if preserved
    inputs_ = []
    indexed_inputs = dict()
    for i, val in enumerate(block_arg_types.items()):
        inputs_.append(val[1])
        indexed_inputs[val[0]] = i

    outputs_ = []
    output_ops = []
    for i, val in enumerate(ret_arg_types.items()):
        outputs_.append(val[1])
        output_ops.append(val[0])

    module = Module.create(loc=Location.unknown(context))
    with InsertionPoint(module.body), Location.unknown(context):
        # Operations can be created in a generic way.
        f_type = FunctionType.get(inputs_, outputs_)
        fc = func.FuncOp("main", f_type)

        old_to_new_op = dict()

        with InsertionPoint(fc.add_entry_block()):
            args = fc.entry_block.arguments
            with InsertionPoint(fc.regions[0].blocks[0]), Location.unknown(context):
                for op_idx, old_op in enumerate(block):
                    new_op = old_op.clone()

                    for i, r in enumerate(new_op.results):
                        old_to_new_op[old_op.results[i]] = r

                    for i, operand in enumerate(old_op.operands):
                        operand_name = operand
                        if operand_name in old_to_new_op.keys():
                            new_op.operands[i] = old_to_new_op[operand_name]
                        elif operand_name in block_arg_types.keys():
                            new_op.operands[i] = args[
                                indexed_inputs[operand_name]
                            ]

            # this is the return statement?
            if "return" not in block[-1].name:
                # we need a return statement
                ret_val = []
                for op in output_ops:
                    ret_val.append(old_to_new_op[op])
                func.ReturnOp(ret_val)

        # Add private functions
        if private_funcs is not None:
            module = append_private_functions(private_funcs, module, context=context)

    return module, inputs_, outputs_
