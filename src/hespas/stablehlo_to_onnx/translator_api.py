# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
"""
Public API for StableHLO-to-ONNX translation.

Input contract
--------------
The translation entry points in this module operate on a sequence of parsed StableHLO
operations, typically `OpInfo` objects produced by `hespas.mlir_parser.MLIRParser`.

Each op is expected to provide, at minimum:

- `op_name`: StableHLO op name such as `stablehlo.add`
- `input_types`: input tensor metadata as `(shape, dtype)` tuples
- `output_types`: output tensor metadata as `(shape, dtype)` tuples
- `operands`: SSA operand ids
- `results`: SSA result ids

SSA wiring is required. The translator builds graph connectivity from `operands` and
`results`, so callers must pass ops with stable SSA names such as `%arg0` and `%1`.

Fallback behavior
-----------------
If `do_fallback=True`, unsupported ops do not immediately fail translation. Instead,
the translator emits a structural placeholder by reusing the first operand when possible
or synthesizing zero-valued initializers for operand-less ops. This preserves graph
shape well enough for inspection or tooling experiments, but it is not semantically
equivalent to the original StableHLO op.

If `do_fallback=False`, translation fails fast with `ValueError` when an unsupported op
is encountered.

Use `stablehlo_ops_to_onnx_model()` when you need an in-memory `onnx.ModelProto`.
Use `stablehlo_ops_to_onnx_file()` when you want the translated model written directly
to disk.
"""

from typing import Any, Sequence

import onnx_ir as ir
import onnx

from .translator import StableHLOToOnnxTranslator, TranslationOptions


def stablehlo_ops_to_onnx_model(
    ops: Sequence[Any],
    *,
    model_name: str = "stablehlo_model",
    do_fallback: bool = True,
) -> onnx.ModelProto:
    """
    Translate parsed StableHLO ops into an in-memory ONNX model.

    Args:
        ops: Sequence of parsed StableHLO ops. Each op must expose `op_name`,
            `input_types`, `output_types`, `operands`, and `results`. SSA operand and
            result ids are required so the translator can build graph edges.
        model_name: Name assigned to the produced ONNX model.
        do_fallback: If `True`, unsupported ops are lowered using the translator's
            structural fallback path. If `False`, unsupported ops raise `ValueError`.

    Returns:
        onnx.ModelProto: The translated ONNX model.

    Raises:
        ValueError: If an op is malformed, SSA wiring is missing, or an unsupported op
            is encountered while `do_fallback=False`.
        NotImplementedError: If a known lowering rejects an unsupported sub-case.
    """
    tr = StableHLOToOnnxTranslator(
        options=TranslationOptions(
            model_name=model_name,
            do_fallback=do_fallback,
        )
    )
    return ir.to_proto(tr.translate(ops))


def stablehlo_ops_to_onnx_file(
    ops: Sequence[Any],
    out_path: str,
    *,
    model_name: str = "stablehlo_model",
    do_fallback: bool = True,
) -> None:
    """
    Translate parsed StableHLO ops and write the ONNX model to disk.

    This is the file-writing companion to `stablehlo_ops_to_onnx_model()`. Use it when
    you do not need the in-memory `onnx.ModelProto` and want the output serialized
    directly to `out_path`.

    Args:
        ops: Sequence of parsed StableHLO ops. Each op must expose `op_name`,
            `input_types`, `output_types`, `operands`, and `results`. SSA operand and
            result ids are required so the translator can build graph edges.
        out_path: Destination path for the serialized ONNX model.
        model_name: Name assigned to the produced ONNX model.
        do_fallback: If `True`, unsupported ops are lowered using the translator's
            structural fallback path. If `False`, unsupported ops raise `ValueError`.

    Raises:
        ValueError: If an op is malformed, SSA wiring is missing, or an unsupported op
            is encountered while `do_fallback=False`.
        NotImplementedError: If a known lowering rejects an unsupported sub-case.
        OSError: If the output file cannot be written.
    """

    tr = StableHLOToOnnxTranslator(
        options=TranslationOptions(
            model_name=model_name,
            do_fallback=do_fallback,
        )
    )

    ir_model = tr.translate(ops)
    ir.save(ir_model, out_path)
