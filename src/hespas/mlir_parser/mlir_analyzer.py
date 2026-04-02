# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import argparse
from collections import Counter
import base64
import hashlib
import json
from functools import lru_cache, cached_property

from .mlir_parser import MLIRParser
from ..utils.op_info import OpInfo
from ..utils.logging import get_log_levels, logger_basic_config

class MLIRAnalyzer(MLIRParser):
    """
    A class for analyzing MLIR (Multi-Level Intermediate Representation) code.
    It can count unique operators, total operators, and generate a histogram of operator frequencies.
    """

    def __init__(self, *, mlir_string=None, mlir_path=None, mlir_module=None, input_sensitive=False, format='table'):
        """
        Initializes the MLIRAnalyzer with an MLIR string or file path.

        Args:
            mlir_input (str): The MLIR code as a string or the path to an MLIR file.
            input_sensitive (bool): Consider the inputs as part of the operator while considering uniqueness
            format (str): Output format for printing results ('table' or 'csv').
        """
        super().__init__(mlir_string=mlir_string, mlir_path=mlir_path, mlir_module=mlir_module, input_sensitive=input_sensitive)
        self.format = format

    def count_unique_operators(self, expand_func_calls=True):
        """
        Counts the number of unique operators in the MLIR code.

        Args:
            expand_func_calls (bool): Whether to use expanded operators or not

        Returns:
            int: The number of unique operators.
        """
        operators = self.operators_expanded if expand_func_calls else self.operators
        return len(set(operators))

    def total_number_of_operators(self, expand_func_calls=True):
        """
        Counts the total number of operators in the MLIR code.

        Returns:
            int: The total number of operators.
        """
        return self.op_count_expanded if expand_func_calls else self.op_count

    def histogram_of_operators(self, expand_func_calls=True):
        """
        Generates a histogram (dictionary) of operator frequencies.

        Returns:
            dict: A dictionary where keys are operators and values are their counts.
        """
        ops = self.operators_expanded if expand_func_calls else self.operators
        return dict(Counter(ops))

    def update(self, other_analyzer):
        """
        Updates the current analyzer's operator list with operators from another MLIRAnalyzer instance.

        Args:
            other_analyzer (MLIRAnalyzer): Another MLIRAnalyzer instance.
        """
        if not isinstance(other_analyzer, MLIRAnalyzer):
            raise TypeError("other_analyzer must be an instance of MLIRAnalyzer.")
        self.operators.extend(other_analyzer.operators)

    @cached_property
    def op_count(self):
        return len(self.operators)

    @cached_property
    def op_count_expanded(self):
        return len(self.operators_expanded)

    @cached_property
    def op_count_and_expanded(self):
        return self.op_count, self.op_count_expanded

    @property
    def operators(self):
        if not hasattr(self, "_operators"):
            self.get_operator_counts()
        return self._operators

    @property
    def operators_expanded(self):
        if not hasattr(self, "_operators_expanded"):
            self.get_operator_counts()
        return self._operators_expanded

    @staticmethod
    @lru_cache(maxsize=10240, typed=False)
    def get_op_key(operation, input_sensitive):
        if isinstance(operation, OpInfo):
            if input_sensitive is False:
                return operation.op_name
            else:
                return json.dumps({operation.op_name: [str(inp) for inp in operation.inputs]})
        elif input_sensitive is False:
            return str(operation.name)
        else:
            return json.dumps({str(operation.name): [str(arg.type) for arg in operation.operands]})

    def get_operator_counts(self):
        self._operators_expanded = []
        self._operators = []
        for operation in self.operations:
            operator_name = MLIRAnalyzer.get_op_key(operation, self.input_sensitive)
            self._operators.append(operator_name)
            self._operators_expanded.extend(self.get_expanded_operator_counts(operation, operator_name))

    def get_expanded_operator_counts(self, operation, operator_name):
        expanded_ops = []

        if operator_name == "func.call":
            # Get the function name being called
            if "callee" in operation.attributes:
                callee_name = operation.attributes["callee"].value
                if isinstance(callee_name, tuple):
                    callee_name = callee_name[-1]

                # Get the private function map
                func_map = self.get_private_functions_map()

                if callee_name in func_map:
                    # Recursively get operations from the called function
                    expanded_ops.extend(self.get_expanded_operator_counts_list(self.get_private_functions_ops(func_map[callee_name])))
                else:
                    # If function not found, just add the func.call itself
                    expanded_ops.append(operator_name)
            else:
                # If no callee attribute, just add the func.call itself
                expanded_ops.append(operator_name)
        elif operator_name == "mhlo.fusion":
            for fused_block in operation.fused_computation:
                expanded_ops.extend(self.get_expanded_operator_counts_list(fused_block.operations))
        else:
            # For non-function-call operations, add them directly
            expanded_ops.append(operator_name)

        return expanded_ops

    def get_expanded_operator_counts_list(self, operations):
        expanded_ops = []
        for operation in operations:
            operator_name = MLIRAnalyzer.get_op_key(operation, self.input_sensitive)
            expanded_ops.extend(self.get_expanded_operator_counts(operation, operator_name))
        return expanded_ops

    def to_str(self, expand = True):
        """
        Prints the MLIR statistics in a formatted way.
        """
        # Default prints use expanded (inlined) operations
        unique_operators = self.count_unique_operators(expand_func_calls=expand)
        total_operators = self.total_number_of_operators(expand_func_calls=expand)
        operator_histogram = self.histogram_of_operators(expand_func_calls=expand)

        s = "MLIR Analysis Results:\n"
        sep = "-" * 35 + "\n"
        s += sep
        s += f"Unique Operators   : {unique_operators}\n"
        s += "\nOperator Histogram:\n"
        for operator, count in sorted(
            operator_histogram.items(), key=lambda item: item[1], reverse=True
        ):
            if ":" in operator:
                op = operator[operator.find('"')+1: operator.find(":")]
                hash_ = operator[operator.find(":"):]
                hasher = hashlib.sha1(hash_.encode("utf-8"))
                fin_hash = base64.urlsafe_b64encode(hasher.digest()[:8])
                operator = f'{op}_{fin_hash}'
            if self.format.lower() == 'table':
                s += f"  {operator:.<30}: {count:>5}\n"
            elif self.format.lower() == 'csv':
                s += f"{operator},{count}\n"
            else:
                raise ValueError(f"Unsupported Format {self.format}")
        s += sep
        s += f"  {'Total Operators':.<30}: {total_operators:>5}\n"
        return s

def main():
    """
    Main function to parse command-line arguments and run MLIR analysis.
    """
    parser = argparse.ArgumentParser(description="Analyze MLIR code.")
    parser.add_argument("mlir_file", help="Path to the MLIR file.")
    parser.add_argument("-s", "--input_sensitive", action="store_true", default=False, help="Input sensitive operator matching.")
    parser.add_argument("-f", "--format", default="table", help="table/csv output for print")
    parser.add_argument("-e", "--expand_func_calls", action="store_true", help="Expand function calls in the output.")
    parser.add_argument("--log-path", default=None, type=str, help="Output path for logging")
    parser.add_argument("--log-level", default='info', type=str, choices=get_log_levels(), help="Set log level")
    args = parser.parse_args()
    logger_basic_config(filename=args.log_path, level=args.log_level)

    analyzer = MLIRAnalyzer(mlir_path=args.mlir_file, input_sensitive=args.input_sensitive, format=args.format)
    print(analyzer.to_str(expand=args.expand_func_calls))

if __name__ == "__main__":
    main()
