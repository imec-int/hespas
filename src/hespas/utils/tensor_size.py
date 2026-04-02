# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
from functools import lru_cache
from jax._src.interpreters.mlir import make_ir_context
from .dt_sizes import get_aligned_bytes

@lru_cache
def compute_tensor_size(tensor_type, context=None) -> int:
    if context is None:
        context = make_ir_context()
    context.allow_unregistered_dialects = True
    return get_aligned_bytes((tuple(x for x in tensor_type.shape if x != -1), tensor_type.element_type))

