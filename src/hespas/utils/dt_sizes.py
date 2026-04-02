# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import math
import re
from functools import lru_cache

datatype_bits_re = re.compile(r"([ustb]?[if])([0-9]+)")

@lru_cache
def get_bit_width(dtype):
    if hasattr(dtype, "width"):
        return dtype.width
    elif isinstance(dtype, str):
        match = datatype_bits_re.match(dtype)
        if match:
            return int(match.group(2))
    elif hasattr(dtype, "element_type") and "complex" in str(dtype):
        return 2 * get_bit_width(dtype.element_type)
    raise ValueError("Cannot get the bit width for {} of type {}".format(dtype, type(dtype).__name__))

@lru_cache
def get_tensor_bit_width(tensor_tuple):
    num_elements = math.prod(tensor_tuple[0]) if len(tensor_tuple[0]) > 0 else 1
    bit_width = get_bit_width(tensor_tuple[1])
    return num_elements * bit_width

@lru_cache
def get_aligned_bytes(dtype, alignment=1):
    bits = get_tensor_bit_width(dtype) if isinstance(dtype, (tuple, list)) and len(dtype) == 2 else get_bit_width(dtype)
    return max((bits if bits % (alignment * 8) == 0 else bits + (alignment * 8)) // 8, alignment)
