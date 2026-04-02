# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import jaxlib.mlir as mlir
from jax._src.interpreters.mlir import make_ir_context
from jaxlib.mlir.ir import Module, WalkOrder, WalkResult
from ..utils.op_info import OpInfo
from .mlir_common import get_dot_general_dimensions
import networkx as nx
import logging
import re
from functools import cached_property
log = logging.getLogger(__name__)


class MLIRParser:
    """
    A class for parsing MLIR (Multi-Level Intermediate Representation) code.
    """

    ir_context = None
    conv_dim_number_re = re.compile(r"stablehlo\.conv\s*<\s*(\[\s*([fb01])\s*,\s*([fb01])\s*,\s*([fb01])\s*,\s*([fb01])\s*\])\s*x\s*(\[\s*([io01])\s*,\s*([io01])\s*,\s*([io01])\s*,\s*([io01])\s*\])\s*->\s*(\[\s*([fb01])\s*,\s*([fb01])\s*,\s*([fb01])\s*,\s*([fb01])\s*\])>")
    window_attr_re = re.compile(r"window\s*=\s*{([^}]*)\s*}")
    window_sub_attr_re = re.compile(r"([a-zA-Z_\-0-9]+)\s*=\s*(\[[^=]+\])")

    @staticmethod
    def get_ir_context():
        if MLIRParser.ir_context is None:
            MLIRParser.ir_context = make_ir_context()
        return MLIRParser.ir_context

    @staticmethod
    def get_module(mlir_string, context=None):
        if context is None:
            context = MLIRParser.get_ir_context()
        return Module.parse(mlir_string.strip(), context=context)

    def __init__(self, *, mlir_string=None, mlir_path=None, mlir_module=None, input_sensitive=False):
        """
        Initializes the MLIRParser with an MLIR string or file path.

        Args:
            mlir_input (str): The MLIR code as a string or the path to an MLIR file.
            input_sensitive (bool): Consider the inputs as part of the operator while considering uniqueness
        """
        self.input_sensitive = input_sensitive

        if sum([x is not None for x in [mlir_string, mlir_path, mlir_module]]) > 1:
            raise ValueError("Only one of either mlir_string or mlir_path or mlir_module must be specified")

        if mlir_path is not None:
            with open(mlir_path, "r") as f:
                log.info(f"{type(self).__name__}: Parsing mlir_file: {mlir_path}")
                mlir_string = f.read()

        if mlir_string is not None:
            mlir_module = MLIRParser.get_module(mlir_string)
            self._mlir_string = mlir_string

        if mlir_module is None:
            raise ValueError("One of mlir_string or mlir_path or mlir_module must be passed")

        # Module objects in MLIR Python bindings are tied to the Context they are created in.
        # If the Context is destroyed or goes out of scope, the Module may become invalid.
        # Therefore, you should keep a reference to the Context as long as you need the Module.
        self.module = mlir_module

    @property
    def mlir_string(self):
        if not hasattr(self, "_mlir_string"):
            self._mlir_string = str(self.module)
        return self._mlir_string

    @cached_property
    def operations(self):
        """
        Extracts operators from the MLIR code using regular expressions.
          operation_input: True includes operator inputs in the categorization
        Returns:
            list: A list of extracted operators.
        """
        return self.main_function.regions[0].blocks[0].operations


    @cached_property
    def main_function(self):
        """
        Returns the main function in the MLIR module.
        """
        return self.module.body.operations[self.main_index]


    @cached_property
    def main_index(self) -> int:
        """
        Returns the index of the main function in the MLIR module.
        Raises an error if the main function is not found.
        """
        for i, op in enumerate(self.module.body.operations):
            op_name = op.name
            if hasattr(op_name, "value"):
                op_name = op_name.value
            if op_name == "main":
                return i
        raise ValueError("Main function not found in the MLIR module.")


    @property
    def private_functions(self) -> list[mlir.ir.Operation]:
        """
        Returns a list of private functions in the MLIR module.
        """
        if not hasattr(self, "_private_functions"):
            self._private_functions = list(self.module.body.operations)[:self.main_index] + \
                                       list(self.module.body.operations)[self.main_index + 1:]
        return self._private_functions


    def get_private_functions_map(self, funcs = None) -> dict:
        if funcs is None:
            funcs = self.private_functions
        return {
            f.attributes["sym_name"].value: f
            for f in funcs
            if "sym_name" in f.attributes
        }


    def get_private_functions_nx_tree(self, functions):
        name_to_op = self.get_private_functions_map(functions)
        G = nx.DiGraph()
        G.add_nodes_from(name_to_op.keys())

        for caller_name, f in name_to_op.items():
            def collect_visitor(op):
                if op.name == "func.call":
                    if "callee" in op.attributes:
                        callee = op.attributes["callee"].value
                        if isinstance(callee, tuple):
                            callee = callee[-1]
                        if callee in name_to_op:
                            G.add_edge(caller_name, callee)
                return WalkResult.ADVANCE
            f.operation.walk(collect_visitor, walk_order=WalkOrder.PRE_ORDER)
        return G, name_to_op


    def get_private_functions_ops(self, function):
        """ Returns a list of operations in a private function, including nested calls. """
        ops = []
        def visitor(op):
            if op.name != "func.func" and op.name != "func.return":
                _ = self.parse_operation(op)
                ops.extend(_ if isinstance(_, list) else [_])
            return WalkResult.ADVANCE
        function.operation.walk(visitor, walk_order=WalkOrder.PRE_ORDER)
        return ops

    def parse_operation(self, operation):

        def get_type_list(op_type):
            if "!mhlo.async_bundle" in str(op_type):
                op_type = op_type.parse(str(op_type).replace("!mhlo.async_bundle", "tuple"), context=MLIRParser.get_ir_context())
            ret_list = [op_type]
            if hasattr(op_type, "num_types"):
                ret_list = []
                for i in range(op_type.num_types):
                    ret_list.extend(get_type_list(op_type.get_type(i)))
            return ret_list

        operation = operation.opview
        name = str(operation.name)

        inputs = []
        outputs = []
        operand_ids = []
        result_ids = []
        for arg in operation.operands:
            operand_ids.append("%{}".format(hash(arg)))
            for arg_type in get_type_list(arg.type):
                inputs.append((tuple(arg_type.shape), arg_type.element_type))
        for res in operation.results:
            result_ids.append("%{}".format(hash(res)))
            for res_type in get_type_list(res.type):
                outputs.append((tuple(res_type.shape), res_type.element_type))

        # NOTE: attributes are ignored at the moment
        op_info = OpInfo(name, inputs, outputs, operands=operand_ids, results=result_ids, opview=operation)
        if 'stablehlo.reduce' == name:
            op_info.dimensions = [i for i in operation.attributes["dimensions"]]
        if 'stablehlo.transpose' == name:
            op_info.attributes["permutation"] = [i for i in operation.attributes["permutation"]]
        elif 'stablehlo.dot_general' in name:
            lhs_contracting_dims, rhs_contracting_dims, batch_dims = get_dot_general_dimensions(operation)
            op_info.dims = {'lhs': lhs_contracting_dims, 'rhs': rhs_contracting_dims, 'batch': batch_dims}
            op_info.lhs_dims = operation.lhs.type.shape
            op_info.rhs_dims = operation.rhs.type.shape
        elif 'stablehlo.dot' in name:
            if operation.lhs.type.shape[1] != operation.rhs.type.shape[0] or len(operation.lhs.type.shape) != len(operation.rhs.type.shape) or len(operation.lhs.type.shape) != 2:
                raise ValueError("Assumption of [M, K] . [K, N] = [M, N] is invalid - {}".format(operation))
            op_info.dims = {'lhs': [1]}
            op_info.lhs_dims = operation.lhs.type.shape
            op_info.rhs_dims = operation.rhs.type.shape
        elif 'stablehlo.convolution' in name:
            dim_numbers_match = self.conv_dim_number_re.search(str(operation.dimension_numbers))
            dim_map = None
            if dim_numbers_match:
                dim_map = {"lhs": {}, "rhs": {}, "out": {}}
                for dim_type, offset in zip(list(dim_map.keys()), (0, 5, 10)):
                    for element, dim in enumerate(dim_numbers_match.group(2+offset, 3+offset, 4+offset, 5+offset)):
                        if dim == "0":
                            dim_map[dim_type]["0_dimension"] = element
                        elif dim == "1":
                            dim_map[dim_type]["1_dimension"] = element
                        elif dim == "f" and dim_type in ["lhs", "out"]:
                            dim_map[dim_type]["feature_dimension"] = element
                        elif dim == "b" and dim_type in ["lhs", "out"]:
                            dim_map[dim_type]["batch_dimension"] = element
                        elif dim == "i" and dim_type in ["rhs"]:
                            dim_map[dim_type]["input_feature_dimension"] = element
                        elif dim == "o" and dim_type in ["rhs"]:
                            dim_map[dim_type]["output_feature_dimension"] = element
                        else:
                            raise ValueError("Unexpected dimension type '{}' for type {}".format(dim, dim_type))
            op_info.dimension_numbers = dim_map
            window_attr = self.window_attr_re.search(op_info.op_line)
            if window_attr:
                ret_dict = {}
                res = self.window_sub_attr_re.findall(window_attr.group(1))
                for res_key, res_val in res:
                    ret_dict[res_key.strip()] = eval(res_val.replace("false", "False").replace("true", "True"))
                op_info.attributes["window"] = ret_dict
            op_info.feature_group_count = int(operation.attributes["feature_group_count"]) if "feature_group_count" in operation.attributes else 1
        elif 'func.call' in name:
            fname = operation.attributes["callee"].value
            fmap = self.get_private_functions_map()
            return self.get_private_functions_ops(fmap[fname])
        elif "stablehlo.custom_call" in name:
            if hasattr(operation, "call_target_name"):
                op_info.kernel_name = str(operation.call_target_name)
            else:
                log.warning("Custom call doesn't have a call_target")
                op_info.kernel_name = ""
        elif "mhlo.fusion" in name:
            if hasattr(operation, "fusion_kind"):
                op_info.fusion_kind = operation.fusion_kind
            elif "fusion_kind" in operation.attributes:
                op_info.fusion_kind = str(operation.attributes["fusion_kind"])
            else:
                op_info.fusion_kind = None
                log.warning("Fusion operation has no fusion_kind")
            op_info.fused_ops = []
            if hasattr(operation, "fused_computation"):
                for fused_block in operation.fused_computation.blocks:
                    for fused_op in fused_block.operations:
                       op_info.fused_ops.append(self.parse_operation(fused_op))
            elif hasattr(operation, "block"):
                for fused_op in operation.block.operations:
                    op_info.fused_ops.append(self.parse_operation(fused_op))
            else:
                log.warning("Fusion operation has no fused_computation: {}".format(operation))
        elif "mhlo.async_start" in name:
            op_info.async_called_computation = str(operation.called_computation)
        elif "mhlo.async_done" in name:
            op_info.async_called_computation = str(operation.attributes["called_computation"])
        elif "stablehlo.constant" in name:
            op_info.value = operation.value
        elif "stablehlo.broadcast_in_dim" in name:
            op_info.broadcast_dimensions = [i for i in operation.broadcast_dimensions] if hasattr(operation, "broadcast_dimensions") else None
        elif "stablehlo.concatenate" in name:
            op_info.dimension = int(operation.dimension) if hasattr(operation, "dimension") else None
        elif "stablehlo.slice" in name:
            op_info.start_indices = [int(i) for i in operation.start_indices] if hasattr(operation, "start_indices") else None
            op_info.limit_indices = [int(i) for i in operation.limit_indices] if hasattr(operation, "limit_indices") else None
            op_info.strides = [int(i) for i in operation.strides] if hasattr(operation, "strides") else None
        elif "stablehlo.compare" in name:
            ct = str(operation.compare_type)
            cd = str(operation.comparison_direction)

            ct = ct.lstrip("#stablehlo<comparison_type ").rstrip(">")
            cd = cd.lstrip("#stablehlo<comparison_direction ").rstrip(">")

            op_info.compare_type = ct
            op_info.comparison_direction = cd

        elif "stablehlo.reduce_window" in name or "stablehlo.select_and_scatter" in name:
            op_info.window_dimensions = [int(i) for i in operation.window_dimensions] if hasattr(operation, "window_dimensions") else None
            op_info.window_strides = [int(i) for i in operation.window_strides] if hasattr(operation, "window_strides") else None
            op_info.base_dilations = [int(i) for i in operation.base_dilations] if hasattr(operation, "base_dilations") else None
            op_info.window_dilations = [int(i) for i in operation.window_dilations] if hasattr(operation, "window_dilations") else None
            if hasattr(operation, "padding"):
                raw_padding = [int(i) for i in operation.padding]
                op_info.padding = [(raw_padding[i], raw_padding[i + 1]) for i in range(0, len(raw_padding), 2)]
            else:
                op_info.padding = None

        elif "stablehlo.reduce_precision" in name:
            op_info.exponent_bits = int(operation.exponent_bits) if hasattr(operation, "exponent_bits") else None
            op_info.mantissa_bits = int(operation.mantissa_bits) if hasattr(operation, "mantissa_bits") else None
        return op_info

    @cached_property
    def ops_list(self):
        """
        Reads a mlir file or mlir_str, returns a list of <operator_name, inputs and output> tuples
        """
        op_list = []
        for operation in self.operations:
            op_info = self.parse_operation(operation)
            op_list.extend(op_info if isinstance(op_info, list) else [op_info])

        if op_list == []:
            log.warning("Function does not have operands.")
        return op_list


    def get_mlir_function_inputs(self):
        """Reads an MLIR file and returns a list of function input types."""

        inputs = []

        if not isinstance(self.module.body.operations[0], mlir.dialects.func.FuncOp):
            raise ValueError("First operation must be a function, not '{}'".format(self.module.body.operations[0]))
        for arg in self.module.body.operations[0].arguments:
            inputs.append(arg.type)
        if inputs == []:
            log.warning("Function does not have inputs")
        return inputs


    def get_mlir_function_outputs(self):
        """Reads an MLIR file and returns a list of function output types."""

        ret = []

        if not isinstance(self.module.body.operations[0], mlir.dialects.func.FuncOp):
            raise ValueError("First operation must be a function, not '{}'".format(self.module.body.operations[0]))
        ret = self.module.body.operations[0].type.results
        if ret == []:
            log.warning("Function does not have return value.")
        return ret
