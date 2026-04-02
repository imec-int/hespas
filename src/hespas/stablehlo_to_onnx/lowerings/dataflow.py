# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
from onnx import numpy_helper
import onnx_ir as ir
from onnx_ir import Attr, AttributeType
import re

from ..registry import LoweringRegistry

TYPE_MAP = {
        ir.DataType.FLOAT: np.float32,
        ir.DataType.DOUBLE: np.float64,
        ir.DataType.FLOAT16: np.float16,
        ir.DataType.INT64: np.int64,
        ir.DataType.INT32: np.int32,
        ir.DataType.INT16: np.int16,
        ir.DataType.INT8: np.int8,
        ir.DataType.UINT64: np.uint64,
        ir.DataType.UINT32: np.uint32,
        ir.DataType.UINT16: np.uint16,
        ir.DataType.UINT8: np.uint8,
        ir.DataType.BOOL: np.bool_,
    }

def register_dataflow_lowerings(reg: LoweringRegistry) -> None:
    reg.register("stablehlo.constant", lower_constant)
    reg.register("stablehlo.convert", lower_convert)
    reg.register("stablehlo.broadcast_in_dim", lower_broadcast_in_dim)
    reg.register("stablehlo.concatenate", lower_concatenate)
    reg.register("stablehlo.select", lower_select)
    reg.register("stablehlo.slice", lower_slice)
    reg.register("stablehlo.compare", lower_compare)


def _constant_initializer_name(tr: Any, out_decl: Any) -> str:
    return tr.fresh_value_name()


def lower_constant(tr: Any, op: Any) -> None:
    """
    Lower stablehlo.constant to an ONNX initializer (preferred) or Constant node.
    The output Value (declared) must exist via tr.get_results(op).
    """
    outs = tr.get_results(op)
    if len(outs) != 1:
        raise ValueError("stablehlo.constant expects 0 inputs -> 1 output")
    out_decl = outs[0]

    dtype = tr.dtype_of(out_decl)
    shape = tr.shape_of(out_decl)

    raw = getattr(op, "value", None)

    if raw is None:
        raise ValueError("stablehlo.constant: could not find constant payload on OpInfo")

    arr = None
    if isinstance(raw, np.ndarray):
        arr = raw

    tname = type(raw).__name__
    if isinstance(raw, np.ndarray):
        pass
    elif tname in ("DenseFPElementsAttr", "DenseIntElementsAttr", "DenseElementsAttr"):
        # Splat: single repeated value
        if raw.is_splat:
            val = raw.get_splat_value().value
            arr = np.asarray(val)
        elif dtype == ir.DataType.BFLOAT16:
            arr = None
        elif dtype in TYPE_MAP:
            arr = np.asarray(raw)
        else:
            raise ValueError(
                f"stablehlo.constant: could not convert dtype {dtype} to numpy array for payload type {tname}"
            )
    else:
        raise ValueError(f"stablehlo.constant: unsupported constant payload type: {tname}")

    if dtype == ir.DataType.BFLOAT16:
        def _parse_dense_floats_from_attr(raw_attr: Any) -> list[float]:
            s = str(raw_attr)
            start = s.find("dense<")
            if start == -1:
                raise ValueError("stablehlo.constant: unsupported bfloat16 payload format")
            start += len("dense<")
            end = s.find("> :", start)
            if end == -1:
                end = s.find(">:", start)
            if end == -1:
                raise ValueError("stablehlo.constant: unsupported bfloat16 payload format")
            payload = s[start:end].strip()
            if payload.startswith("[") and payload.endswith("]"):
                payload = payload[1:-1]
            nums = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", payload)
            if not nums:
                raise ValueError("stablehlo.constant: no numeric values found for bfloat16 payload")
            return [float(x) for x in nums]


        if raw.is_splat:
            val = raw.get_splat_value().value
            if shape is None:
                a32 = np.asarray([val], dtype=np.float32)
                dims = []
            else:
                dims = [int(d) for d in shape]
                a32 = np.full(dims, val, dtype=np.float32)
        else:
            vals = _parse_dense_floats_from_attr(raw)
            a32 = np.asarray(vals, dtype=np.float32)
            if shape is not None:
                dims = [int(d) for d in shape]
                try:
                    a32 = a32.reshape(tuple(dims))
                except ValueError as exc:
                    raise ValueError(
                        f"stablehlo.constant: could not reshape constant to declared shape {shape}"
                    ) from exc
            else:
                dims = list(a32.shape)

        u32 = a32.view(np.uint32)
        u16 = (u32 >> 16).astype(np.uint16)

        raw_data = numpy_helper.tobytes_little_endian(u16)

        tp = onnx.helper.make_tensor(
            _constant_initializer_name(tr, out_decl),
            onnx.TensorProto.BFLOAT16,
            dims,
            raw_data,
            raw=True,
        )
        produced = ir.Value(
            name=tp.name,
            type=ir.TensorType(dtype),
            shape=(out_decl.shape),
            const_value=ir.tensor(tp),
        )
        tr.builder.initializers.append(produced)
        tr.bind_result_value(out_decl, produced)
        return

    if arr is None:
        raise ValueError(f"stablehlo.constant: could not convert dtype {dtype} to numpy array for payload type {tname}")

    if shape is not None:
        try:
            arr = np.asarray(arr).reshape(tuple(int(d) for d in shape))
        except ValueError as exc:
            raise ValueError(
                f"stablehlo.constant: could not reshape constant to declared shape {shape}"
            ) from exc

    target_np = TYPE_MAP.get(dtype)
    if target_np is None:
        raise TypeError(f"Unsupported IR dtype for constant: {dtype}")
        
    arr = np.asarray(arr, dtype=target_np)
    produced_name = _constant_initializer_name(tr, out_decl)
    produced = ir.Value(
        name=produced_name,
        type=ir.TensorType(dtype),
        shape=(out_decl.shape),
        const_value=ir.tensor(arr),
    )
    tr.builder.initializers.append(produced)
    tr.bind_result_value(out_decl, produced)

def lower_convert(tr: Any, op: Any) -> None:
    """
    stablehlo.convert -> ONNX Cast

    Uses result dtype as the Cast target.
    """
    ins = tr.get_operands(op)
    outs = tr.get_results(op)
    if len(ins) != 1 or len(outs) != 1:
        raise ValueError("stablehlo.convert expects 1 input -> 1 output")

    x = ins[0]
    y_decl = outs[0]

    to_dtype = tr.dtype_of(y_decl)
    to_onnx = to_dtype.value if isinstance(to_dtype, ir.DataType) else 1

    y = tr.make_temp(dtype=to_dtype, shape=tr.shape_of(y_decl), name_hint="convert")
    tr.builder.add_node(
        op_type="Cast",
        inputs=[x],
        outputs=[y],
        attributes=[Attr("to", AttributeType.INT, int(to_onnx))],
        name=tr.fresh_node_name("Cast"),
    )
    tr.bind_result_value(y_decl, y)

def lower_broadcast_in_dim(tr: Any, op: Any) -> None:
    """
    stablehlo.broadcast_in_dim -> Unsqueeze + Expand

    Expected:
      - 1 operand
      - broadcast_dimensions: list[int] mapping input dims -> output dims

    We'll:
      1) unsqueeze at output axes not present in broadcast_dimensions
      2) Expand to the result shape
    """
    ins = tr.get_operands(op)
    outs = tr.get_results(op)
    if len(ins) != 1 or len(outs) != 1:
        raise ValueError("stablehlo.broadcast_in_dim expects 1 input -> 1 output")

    x = ins[0]
    y_decl = outs[0]

    out_shape = tr.shape_of(y_decl)
    if out_shape is None:
        raise ValueError("broadcast_in_dim: output shape is required")

    bd = op.broadcast_dimensions

    if not isinstance(bd, (list, tuple)):
        raise TypeError("broadcast_in_dim missing broadcast_dimensions list[int]")
    bd = [int(d) for d in bd]

    in_rank = tr.rank_of(x)
    out_rank = len(out_shape)
    if in_rank is None:
        # Still can proceed if bd is consistent.
        in_rank = len(bd)

    if len(bd) != in_rank:
        raise ValueError(
            f"broadcast_in_dim: broadcast_dimensions len={len(bd)} does not match input rank={in_rank}"
        )

    # Normalize negative indices
    bd = [(d + out_rank) if d < 0 else d for d in bd]
    if any(d < 0 or d >= out_rank for d in bd):
        raise ValueError("broadcast_in_dim: broadcast_dimensions out of range")

    missing_axes = [ax for ax in range(out_rank) if ax not in set(bd)]

    x2 = x
    if missing_axes:
        axes_init = tr.builder.const_i64(
            tr.fresh_value_name(),
            [int(a) for a in missing_axes],
        )

        unsq = tr.make_temp(
            dtype=tr.dtype_of(x),
            shape=None,
            name_hint="unsqueeze",
        )

        tr.builder.add_node(
            op_type="Unsqueeze",
            inputs=[x2, axes_init],  # <-- axes is now INPUT
            outputs=[unsq],
            attributes=[],
            name=tr.fresh_node_name("Unsqueeze"),
        )

        x2 = unsq

    # Expand shape tensor
    expand_shape_vals = [int(d) if isinstance(d, (int, np.integer)) else -1 for d in out_shape]
    shape_init = tr.builder.const_i64(tr.fresh_value_name(), expand_shape_vals)

    y = tr.make_temp(dtype=tr.dtype_of(x), shape=list(out_shape), name_hint="broadcast")
    tr.builder.add_node(
        op_type="Expand",
        inputs=[x2, shape_init],
        outputs=[y],
        attributes=[],
        name=tr.fresh_node_name("Expand"),
    )
    tr.bind_result_value(y_decl, y)


def lower_concatenate(tr: Any, op: Any) -> None:
    """
    stablehlo.concatenate -> ONNX Concat
    Expects:
      - N operands
      - attribute 'dimension' (axis)
    """
    ins = tr.get_operands(op)
    outs = tr.get_results(op)
    if len(ins) < 1 or len(outs) != 1:
        raise ValueError("stablehlo.concatenate expects N inputs -> 1 output")

    axis = op.dimension

    if axis is None:
        raise TypeError("stablehlo.concatenate missing 'dimension' attribute")
    axis = int(axis)

    rank = tr.rank_of(ins[0])
    if rank is not None and axis < 0:
        axis = axis + rank

    y_decl = outs[0]
    y = tr.make_temp(dtype=tr.dtype_of(y_decl), shape=tr.shape_of(y_decl), name_hint="concat")
    tr.builder.add_node(
        op_type="Concat",
        inputs=list(ins),
        outputs=[y],
        attributes=[Attr("axis", AttributeType.INT, int(axis))],
        name=tr.fresh_node_name("Concat"),
    )
    tr.bind_result_value(y_decl, y)


def lower_select(tr: Any, op: Any) -> None:
    """
    stablehlo.select -> ONNX Where

    Expects:
      3 operands:
        0: predicate (bool tensor)
        1: on_true
        2: on_false

      1 output
    """
    ins = tr.get_operands(op)
    outs = tr.get_results(op)

    if len(ins) != 3 or len(outs) != 1:
        raise ValueError("stablehlo.select expects 3 inputs -> 1 output")

    cond, x, y = ins
    out_decl = outs[0]

    cond_dtype = tr.dtype_of(cond)
    if cond_dtype != ir.DataType.BOOL:
        cond_cast = tr.make_temp(
            dtype=ir.DataType.BOOL,
            shape=tr.shape_of(cond),
            name_hint="select_cond_bool",
        )
        tr.builder.add_node(
            op_type="Cast",
            inputs=[cond],
            outputs=[cond_cast],
            
            attributes=[Attr("to", AttributeType.INT, 9)],  # BOOL = 9
            name=tr.fresh_node_name("Cast"),
        )
        cond = cond_cast

    out = tr.make_temp(
        dtype=tr.dtype_of(out_decl),
        shape=tr.shape_of(out_decl),
        name_hint="select",
    )

    tr.builder.add_node(
        op_type="Where",
        inputs=[cond, x, y],
        outputs=[out],
        attributes=[],
        name=tr.fresh_node_name("Where"),
    )

    tr.bind_result_value(out_decl, out)


def lower_slice(tr: Any, op: Any) -> None:
    """
    stablehlo.slice -> ONNX Slice

    StableHLO slice is defined by:
      start_indices: per-dim start (inclusive)
      limit_indices: per-dim end (exclusive)
      strides: per-dim stride (default 1)

    ONNX Slice expects tensors:
      starts, ends, axes, steps

    We emit Slice with:
      axes = [0..rank-1]
      starts = start_indices
      ends = limit_indices
      steps = strides (or 1s)
    """
    ins = tr.get_operands(op)
    outs = tr.get_results(op)
    if len(ins) != 1 or len(outs) != 1:
        raise ValueError("stablehlo.slice expects 1 input -> 1 output")

    x = ins[0]
    y_decl = outs[0]

    rank = tr.rank_of(x)
    if rank is None:
        # we can still proceed if indices lengths are provided, but rank is preferred
        rank = len(tr.shape_of(y_decl) or [])

    start = op.start_indices
    limit = op.limit_indices
    strides = op.strides

    if not all(isinstance(vals, (list, tuple)) for vals in (start, limit, strides)):
        raise TypeError("stablehlo.slice requires start_indices, limit_indices, and strides list[int]")

    if not (len(start) == len(limit) == len(strides)):
        raise ValueError(
            f"stablehlo.slice: start/limit/strides length mismatch: "
            f"{len(start)}/{len(limit)}/{len(strides)}"
        )

    if rank is not None and len(start) != rank:
        raise ValueError(
            f"stablehlo.slice: expected indices length == rank ({rank}), got {len(start)}"
        )

    axes = list(range(len(start)))

    # Build constant tensors for Slice inputs
    starts_init = tr.builder.const_i64(tr.fresh_value_name(), [int(v) for v in start])
    ends_init   = tr.builder.const_i64(tr.fresh_value_name(), [int(v) for v in limit])
    axes_init   = tr.builder.const_i64(tr.fresh_value_name(), [int(a) for a in axes])
    steps_init  = tr.builder.const_i64(tr.fresh_value_name(), [int(s) for s in strides])

    y = tr.make_temp(dtype=tr.dtype_of(y_decl), shape=tr.shape_of(y_decl), name_hint="slice")
    tr.builder.add_node(
        op_type="Slice",
        inputs=[x, starts_init, ends_init, axes_init, steps_init],
        outputs=[y],
        attributes=[],
        name=tr.fresh_node_name("Slice"),
    )
    tr.bind_result_value(y_decl, y)


def lower_compare(tr: Any, op: Any) -> None:
    """
    stablehlo.compare (EQ/LT/LE/GT/GE/NE) -> ONNX Equal/Less/Greater/... (+Not for NE)
    Output is bool (xi1).
    """
    ins = tr.get_operands(op)
    outs = tr.get_results(op)
    if len(ins) != 2 or len(outs) != 1:
        raise ValueError("stablehlo.compare expects 2 inputs -> 1 output")

    a, b = ins
    y_decl = outs[0]

    direction = op.comparison_direction
    direction = direction.strip().upper()

    if direction is None:
        raise ValueError("stablehlo.compare: could not determine comparison direction (EQ/LT/...)")

    direct_op_map = {
        "EQ": "Equal",
        "LT": "Less",
        "LE": "LessOrEqual",
        "GT": "Greater",
        "GE": "GreaterOrEqual",
    }

    if direction in direct_op_map:
        onnx_op = direct_op_map[direction]
        tr.builder.add_node(onnx_op, [a, b], [y_decl], [], tr.fresh_node_name(onnx_op))
        return

    if direction == "NE":
        # NotEqual(a,b) = Not(Equal(a,b))
        eq_tmp = tr.make_temp(dtype=ir.DataType.BOOL, shape=tr.shape_of(y_decl), name_hint="eq")
        tr.builder.add_node("Equal", [a, b], [eq_tmp], [], tr.fresh_node_name("Equal"))
        tr.builder.add_node("Not", [eq_tmp], [y_decl], [], tr.fresh_node_name("Not"))
        return

    raise NotImplementedError(f"stablehlo.compare direction not supported: {direction}")
