# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
from functools import lru_cache
from .dt_sizes import get_bit_width, get_aligned_bytes

class OpInfo:
    def __init__(self, op_name, input_types, output_types, attributes=None, operands=None, results=None, opview=None):
        self.op_name = op_name
        self.input_types = input_types
        self.output_types = output_types
        self.attributes = attributes or {}
        self.operands = operands if operands is not None else []
        self.results = results if results is not None else []
        self.opview = opview

    def __str__(self):
        inputs = ' '.join('x'.join(map(str, dim)) + f'x{t}' for dim, t in self.input_types)
        outputs = ' '.join('x'.join(map(str, dim)) + f'x{t}' for dim, t in self.output_types)
        return f"{self.op_name} ({inputs}) -> ({outputs})"

    @lru_cache
    def get_input_bytes(self, index):
        return get_aligned_bytes(self.input_types[index])

    def get_number_of_inputs(self):
        return len(self.input_types)

    @lru_cache
    def get_output_bytes(self, index):
        return get_aligned_bytes(self.output_types[index])

    @lru_cache
    def get_input_types(self):
        return {x[1] for x in self.input_types}

    @lru_cache
    def get_output_types(self):
        return {x[1] for x in self.output_types}

    @lru_cache
    def get_types(self):
        return self.get_input_types() | self.get_output_types()

    @lru_cache
    def get_largest_type(self):
        return sorted(list(self.get_types()), key=lambda x: get_bit_width(x), reverse=True)[0]

    @property
    def op_line(self):
        if not hasattr(self, "_op_line"):
            self._op_line = str(self.opview)
        return self._op_line

    @op_line.setter
    def op_line(self, value):
        self._op_line = value

    @op_line.deleter
    def op_line(self):
        del self._op_line

    def has_ssa(self):
        return len(self.operands) > 0 and len(self.results) > 0

