# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
from typing import Any, Optional, Sequence
import warnings

import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir

TYPE_MAP = {
    "i1": ir.DataType.BOOL,
    "i8": ir.DataType.INT8,
    "i16": ir.DataType.INT16,
    "i32": ir.DataType.INT32,
    "i64": ir.DataType.INT64,
    "ui8": ir.DataType.UINT8,
    "ui16": ir.DataType.UINT16,
    "ui32": ir.DataType.UINT32,
    "ui64": ir.DataType.UINT64,
    "f16": ir.DataType.FLOAT16,
    "f32": ir.DataType.FLOAT,
    "f64": ir.DataType.DOUBLE,
}

if hasattr(ir.DataType, "BFLOAT16"):
    TYPE_MAP["bf16"] = ir.DataType.BFLOAT16
    TYPE_MAP["bfloat16"] = ir.DataType.BFLOAT16
else:
    TYPE_MAP["bf16"] = ir.DataType.FLOAT16
    TYPE_MAP["bfloat16"] = ir.DataType.FLOAT16

def stablehlo_dtype_to_ir(dtype: Any, *, enable_double_precision: bool) -> ir.DataType:
    """Convert StableHLO dtype tag / MLIR type object to onnx_ir.DataType."""

    fallback = TYPE_MAP['f64'] if enable_double_precision else TYPE_MAP['f32']
    fallback_name = "f64" if enable_double_precision else "f32"
    if dtype is None:
        warnings.warn(
            f"Missing StableHLO dtype; falling back to {fallback_name}",
            stacklevel=2,
        )
        return fallback
    if isinstance(dtype, ir.DataType):
        return dtype

    s = str(dtype).strip().lower()
    np_dt = TYPE_MAP.get(s)

    if np_dt is not None:
        return np_dt

    warnings.warn(
        f"Unrecognized StableHLO dtype '{dtype}'; falling back to {fallback_name}",
        stacklevel=2,
    )
    return fallback


def normalize_shape(shape: Any) -> ir.Shape | None:
    """Convert Sequence[int|str|None] to ir.Shape, preserving explicit unknown dims."""
    if shape is None:
        return None
    if isinstance(shape, ir.Shape):
        return shape
    if isinstance(shape, (list, tuple)):
        dims: list[int | str | None] = []
        for d in shape:
            if d is None:
                dims.append(None)
            elif isinstance(d, (int, np.integer)):
                dims.append(int(d))
            else:
                s = str(d).strip()
                if not s:
                    raise ValueError("Shape dimensions must be ints, None, or non-empty symbolic names")
                dims.append(s)
        return ir.Shape(tuple(dims))  # type: ignore[arg-type]
    return None
