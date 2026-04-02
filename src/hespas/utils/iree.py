# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
from jax._src.interpreters.mlir import make_ir_context

def get_iree_input_string(tensor_type, context=None) -> str:
    if context is None:
        context = make_ir_context()
    context.allow_unregistered_dialects = True
    #TODO: How to handle constants if needed?
    input_str = '--input='
    t_str = str(tensor_type)
    if '<' not in t_str or '>' not in t_str:
        raise ValueError(f"Unsupported input type for iree {tensor_type}")

    t_dim =  t_str[t_str.find('<')+1: t_str.rfind('>')]

    val = '=1'
    # If the input is binary it left empty for IREE to fill default binary inputs.
    if 'i1' in t_dim:
        val = ''

    input_str = f'{input_str}{t_dim}{val}'

    return input_str

