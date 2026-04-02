# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
from typing import Any, List, Sequence

import onnx_ir as ir
from onnx_ir import Attr, AttributeType

from ..registry import LoweringRegistry


def register_reduce_lowerings(reg: LoweringRegistry) -> None:
    reg.register("stablehlo.reduce", lower_reduce)
    reg.register("stablehlo.reduce_window", lower_reduce_window)
    reg.register("stablehlo.reduce_precision", lower_reduce_precision)


def lower_reduce(tr: Any, op: Any) -> None:
    ins = tr.get_operands(op)
    outs = tr.get_results(op)

    # Tuple-reduction form commonly used for argmax(value, index).
    if len(ins) == 4 and len(outs) == 2:
        _lower_reduce_argmax_pair(tr, op, ins, outs)
        return

    if len(ins) != 2 or len(outs) != 1:
        raise ValueError(
            "stablehlo.reduce expects either "
            "2 inputs -> 1 output, or 4 inputs -> 2 outputs (argmax-like tuple reduce). "
            f"op_line: {getattr(op, 'op_line', '')}"
        )

    x, _init = ins
    y_decl = outs[0]

    dims = getattr(op, "dimensions", None)
    if not isinstance(dims, (list, tuple)):
        raise ValueError("stablehlo.reduce missing dimensions/axes list[int]")

    axes = [int(a) for a in dims]
    rank = tr.rank_of(x)
    if rank is not None:
        axes = [(a + rank) if a < 0 else a for a in axes]
        if any(a < 0 or a >= rank for a in axes):
            raise ValueError("stablehlo.reduce: axis out of range")

    op_line = getattr(op, "op_line", "") or ""
    red_key = [k for k in op_line.split() if "stablehlo." in k]

    if "stablehlo.add" in red_key:
        onnx_op = "ReduceSum"
    elif "stablehlo.maximum" in red_key:
        onnx_op = "ReduceMax"
    elif "stablehlo.minimum" in red_key:
        onnx_op = "ReduceMin"
    elif "stablehlo.and" in red_key:
        onnx_op = "ReduceMin"
    elif "stablehlo.or" in red_key:
        onnx_op = "ReduceMax"
    else:
        raise ValueError(f"stablehlo.reduce: could not determine reduction operation from op_line: {op_line}")

    # opset 13+: axes is an INPUT tensor, not an attribute
    axes_init = tr.builder.const_i64(tr.fresh_value_name(), axes)

    y = tr.make_temp(dtype=tr.dtype_of(y_decl), shape=tr.shape_of(y_decl), name_hint="reduce")
    tr.builder.add_node(
        op_type=onnx_op,
        inputs=[x, axes_init],
        outputs=[y],
        attributes=[Attr("keepdims", AttributeType.INT, 0)],
        name=tr.fresh_node_name(onnx_op),
    )

    tr.bind_result_value(y_decl, y)


def _lower_reduce_argmax_pair(tr: Any, op: Any, ins: List[ir.Value], outs: List[ir.Value]) -> None:
    x, idx_in, _init_x, _init_idx = ins
    y_x_decl, y_idx_decl = outs

    dims = getattr(op, "dimensions", None)
    if not isinstance(dims, (list, tuple)):
        raise ValueError("stablehlo.reduce missing dimensions/axes list[int]")
    if len(dims) != 1:
        raise NotImplementedError("tuple stablehlo.reduce currently supports single-axis reduction only")

    axis = int(dims[0])
    rank = tr.rank_of(x)
    if rank is not None and axis < 0:
        axis += rank

    extremum_kind = _infer_tuple_reduce_extremum_kind(getattr(op, "op_line", "") or "")
    if extremum_kind == "max":
        argext_op = "ArgMax"
        argext_hint = "argmax_keep"
    elif extremum_kind == "min":
        argext_op = "ArgMin"
        argext_hint = "argmin_keep"
    else:
        raise ValueError(
            "tuple stablehlo.reduce could not infer arg-extremum kind "
            "(expected compare GT or LT in reducer body)"
        )

    # ArgMax/ArgMin with keepdims=1 gives an index tensor usable by GatherElements.
    argmax_keep = tr.make_temp(dtype=ir.DataType.INT64, shape=None, name_hint=argext_hint)
    tr.builder.add_node(
        op_type=argext_op,
        inputs=[x],
        outputs=[argmax_keep],
        attributes=[
            Attr("axis", AttributeType.INT, int(axis)),
            Attr("keepdims", AttributeType.INT, 1),
            Attr("select_last_index", AttributeType.INT, 0),
        ],
        name=tr.fresh_node_name(argext_op),
    )

    gathered_x_keep = tr.make_temp(dtype=tr.dtype_of(x), shape=None, name_hint="argmax_val_keep")
    tr.builder.add_node(
        op_type="GatherElements",
        inputs=[x, argmax_keep],
        outputs=[gathered_x_keep],
        attributes=[Attr("axis", AttributeType.INT, int(axis))],
        name=tr.fresh_node_name("GatherElements"),
    )

    gathered_idx_keep = tr.make_temp(dtype=tr.dtype_of(idx_in), shape=None, name_hint="argmax_idx_keep")
    tr.builder.add_node(
        op_type="GatherElements",
        inputs=[idx_in, argmax_keep],
        outputs=[gathered_idx_keep],
        attributes=[Attr("axis", AttributeType.INT, int(axis))],
        name=tr.fresh_node_name("GatherElements"),
    )

    squeeze_axes = tr.builder.const_i64(tr.fresh_value_name(), [axis])

    y_x = tr.make_temp(dtype=tr.dtype_of(y_x_decl), shape=tr.shape_of(y_x_decl), name_hint="argmax_val")
    tr.builder.add_node(
        op_type="Squeeze",
        inputs=[gathered_x_keep, squeeze_axes],
        outputs=[y_x],
        attributes=[],
        name=tr.fresh_node_name("Squeeze"),
    )

    y_idx_raw = tr.make_temp(dtype=tr.dtype_of(gathered_idx_keep), shape=tr.shape_of(y_idx_decl), name_hint="argmax_idx")
    tr.builder.add_node(
        op_type="Squeeze",
        inputs=[gathered_idx_keep, squeeze_axes],
        outputs=[y_idx_raw],
        attributes=[],
        name=tr.fresh_node_name("Squeeze"),
    )

    if tr.dtype_of(y_idx_raw) != tr.dtype_of(y_idx_decl):
        y_idx = tr.make_temp(dtype=tr.dtype_of(y_idx_decl), shape=tr.shape_of(y_idx_decl), name_hint="argmax_idx_cast")
        tr.builder.add_node(
            op_type="Cast",
            inputs=[y_idx_raw],
            outputs=[y_idx],
            attributes=[Attr("to", AttributeType.INT, int(tr.dtype_of(y_idx_decl).value))],
            name=tr.fresh_node_name("Cast"),
        )
    else:
        y_idx = y_idx_raw

    tr.bind_result_value(y_x_decl, y_x)
    tr.bind_result_value(y_idx_decl, y_idx)


def _infer_tuple_reduce_extremum_kind(op_line: str) -> str | None:
    # For tuple (value, index) reducers, value compare direction determines:
    # - compare GT => argmax
    # - compare LT => argmin
    for line in op_line.splitlines():
        s = line.strip()
        if not s.startswith("%") or "stablehlo.compare" not in s:
            continue
        if " GT," in s:
            return "max"
        if " LT," in s:
            return "min"
    return None


def lower_reduce_window(tr: Any, op: Any) -> None:
    ins = tr.get_operands(op)
    outs = tr.get_results(op)
    if len(ins) != 2 or len(outs) != 1:
        raise ValueError("stablehlo.reduce_window expects 2 inputs (data, init) -> 1 output")

    x, _init = ins
    y_decl = outs[0]

    x_shape = tr.shape_of(x)
    if x_shape is None:
        raise ValueError("stablehlo.reduce_window requires known input shape")
    rank = len(x_shape)
    if rank < 3:
        raise NotImplementedError("stablehlo.reduce_window currently supports rank >= 3 tensors only")

    wd = [int(v) for v in (getattr(op, "window_dimensions", []))]
    ws = [int(v) for v in (getattr(op, "window_strides", []) or ([1] * rank))]
    bd = [int(v) for v in (getattr(op, "base_dilations", []) or ([1] * rank))]
    wdil = [int(v) for v in (getattr(op, "window_dilations", []) or ([1] * rank))]
    padding = list(getattr(op, "padding", []) or ([(0, 0)] * rank))

    if not (len(wd) == len(ws) == len(bd) == len(wdil) == len(padding) == rank):
        raise ValueError("stablehlo.reduce_window: window attributes must match input rank")

    if any(int(v) != 1 for v in bd):
        raise NotImplementedError("stablehlo.reduce_window with base_dilations != 1 is not supported")
    if any(int(v) != 1 for v in wdil):
        raise NotImplementedError("stablehlo.reduce_window with window_dilations != 1 is not supported")

    non_spatial_axes = [axis for axis in range(rank) if wd[axis] == 1 and ws[axis] == 1]
    if len(non_spatial_axes) != 2:
        raise NotImplementedError(
            "stablehlo.reduce_window currently supports pooling-style windows only "
            "(exactly batch/channel dims unchanged)"
        )

    spatial_axes = [axis for axis in range(rank) if axis not in non_spatial_axes]
    batch_axis = non_spatial_axes[0]
    channel_axis = non_spatial_axes[1]
    kernel_shape = [int(wd[a]) for a in spatial_axes]
    strides = [int(ws[a]) for a in spatial_axes]
    pads = [int(padding[a][0]) for a in spatial_axes] + [int(padding[a][1]) for a in spatial_axes]
    pool_perm = [batch_axis, channel_axis] + spatial_axes
    x_pool = _maybe_transpose_pool_operand(tr, x, pool_perm, name_hint="pool_in")

    y_shape = tr.shape_of(y_decl)
    y_pool_shape = None
    if y_shape is not None:
        y_pool_shape = [y_shape[batch_axis], y_shape[channel_axis]] + [y_shape[a] for a in spatial_axes]

    reducer = _infer_reduce_window_reducer(op.op_line)
    if reducer is None:
        raise ValueError("stablehlo.reduce_window: could not infer reducer kind from op_line")

    if (
        reducer == "add"
        and all(int(wd[a]) == int(x_shape[a]) for a in spatial_axes)
        and all(int(ws[a]) == 1 for a in spatial_axes)
        and all(int(padding[a][0]) == 0 and int(padding[a][1]) == 0 for a in spatial_axes)
    ):
        y_pool = tr.make_temp(dtype=tr.dtype_of(y_decl), shape=y_pool_shape, name_hint="global_avg_pool")
        tr.builder.add_node(
            op_type="GlobalAveragePool",
            inputs=[x_pool],
            outputs=[y_pool],
            attributes=[],
            name=tr.fresh_node_name("GlobalAveragePool"),
        )
        y = _restore_pool_output_layout(tr, y_pool, y_decl, pool_perm, name_hint="global_avg_pool_out")
        tr.bind_result_value(y_decl, y)
        return

    if reducer == "maximum":
        y_pool = tr.make_temp(dtype=tr.dtype_of(y_decl), shape=y_pool_shape, name_hint="maxpool")
        tr.builder.add_node(
            op_type="MaxPool",
            inputs=[x_pool],
            outputs=[y_pool],
            attributes=[
                Attr("kernel_shape", AttributeType.INTS, kernel_shape),
                Attr("strides", AttributeType.INTS, strides),
                Attr("pads", AttributeType.INTS, pads),
            ],
            name=tr.fresh_node_name("MaxPool"),
        )
        y = _restore_pool_output_layout(tr, y_pool, y_decl, pool_perm, name_hint="maxpool_out")
        tr.bind_result_value(y_decl, y)
        return

    if reducer == "minimum":
        x_neg = tr.make_temp(dtype=tr.dtype_of(x_pool), shape=tr.shape_of(x_pool), name_hint="minpool_neg_in")
        tr.builder.add_node(
            op_type="Neg",
            inputs=[x_pool],
            outputs=[x_neg],
            attributes=[],
            name=tr.fresh_node_name("Neg"),
        )

        pooled_neg = tr.make_temp(dtype=tr.dtype_of(x_pool), shape=y_pool_shape, name_hint="minpool_max")
        tr.builder.add_node(
            op_type="MaxPool",
            inputs=[x_neg],
            outputs=[pooled_neg],
            attributes=[
                Attr("kernel_shape", AttributeType.INTS, kernel_shape),
                Attr("strides", AttributeType.INTS, strides),
                Attr("pads", AttributeType.INTS, pads),
            ],
            name=tr.fresh_node_name("MaxPool"),
        )

        y_pool = tr.make_temp(dtype=tr.dtype_of(y_decl), shape=y_pool_shape, name_hint="minpool_out")
        tr.builder.add_node(
            op_type="Neg",
            inputs=[pooled_neg],
            outputs=[y_pool],
            attributes=[],
            name=tr.fresh_node_name("Neg"),
        )
        y = _restore_pool_output_layout(tr, y_pool, y_decl, pool_perm, name_hint="minpool_out_layout")
        tr.bind_result_value(y_decl, y)
        return

    raise NotImplementedError(
        f"stablehlo.reduce_window reducer '{reducer}' is not supported in current pooling lowering"
    )


def _infer_reduce_window_reducer(op_line: str) -> str | None:
    if "stablehlo.maximum" in op_line:
        return "maximum"
    if "stablehlo.minimum" in op_line:
        return "minimum"
    if "stablehlo.add" in op_line:
        return "add"
    return None


def _maybe_transpose_pool_operand(tr: Any, x: ir.Value, perm: Sequence[int], *, name_hint: str) -> ir.Value:
    rank = tr.rank_of(x)
    if rank is None or list(perm) == list(range(rank)):
        return x
    x_shape = tr.shape_of(x)
    y_shape = [x_shape[i] for i in perm] if x_shape is not None else None
    y = tr.make_temp(dtype=tr.dtype_of(x), shape=y_shape, name_hint=name_hint)
    tr.builder.add_node(
        op_type="Transpose",
        inputs=[x],
        outputs=[y],
        attributes=[Attr("perm", AttributeType.INTS, [int(v) for v in perm])],
        name=tr.fresh_node_name("Transpose"),
    )
    return y


def _restore_pool_output_layout(
    tr: Any,
    pooled: ir.Value,
    y_decl: ir.Value,
    pool_perm: Sequence[int],
    *,
    name_hint: str,
) -> ir.Value:
    rank = tr.rank_of(pooled)
    if rank is None or list(pool_perm) == list(range(rank)):
        return pooled

    out_perm = [0] * rank
    for onnx_axis, original_axis in enumerate(pool_perm):
        out_perm[original_axis] = onnx_axis

    y = tr.make_temp(dtype=tr.dtype_of(y_decl), shape=tr.shape_of(y_decl), name_hint=name_hint)
    tr.builder.add_node(
        op_type="Transpose",
        inputs=[pooled],
        outputs=[y],
        attributes=[Attr("perm", AttributeType.INTS, out_perm)],
        name=tr.fresh_node_name("Transpose"),
    )
    return y


def lower_reduce_precision(tr: Any, op: Any) -> None:
    ins = tr.get_operands(op)
    outs = tr.get_results(op)
    if len(ins) != 1 or len(outs) != 1:
        raise ValueError("stablehlo.reduce_precision expects 1 input -> 1 output")

    x = ins[0]
    y_decl = outs[0]

    exp_bits = getattr(op, "exponent_bits", None)
    mant_bits = getattr(op, "mantissa_bits", None)
    if exp_bits is None or mant_bits is None:
        raise ValueError("stablehlo.reduce_precision requires exponent_bits and mantissa_bits")

    target_dtype = _reduce_precision_target_dtype(int(exp_bits), int(mant_bits))
    if target_dtype is None:
        raise NotImplementedError(
            f"stablehlo.reduce_precision({exp_bits}, {mant_bits}) has no direct ONNX cast target"
        )

    x_dtype = tr.dtype_of(x)
    y_dtype = tr.dtype_of(y_decl)

    rounded = x
    if x_dtype != target_dtype:
        rounded_tmp = tr.make_temp(dtype=target_dtype, shape=tr.shape_of(x), name_hint="reduce_precision_lowp")
        tr.builder.add_node(
            op_type="Cast",
            inputs=[x],
            outputs=[rounded_tmp],
            attributes=[Attr("to", AttributeType.INT, int(target_dtype.value))],
            name=tr.fresh_node_name("Cast"),
        )
        rounded = rounded_tmp

    if tr.dtype_of(rounded) != y_dtype:
        y = tr.make_temp(dtype=y_dtype, shape=tr.shape_of(y_decl), name_hint="reduce_precision_out")
        tr.builder.add_node(
            op_type="Cast",
            inputs=[rounded],
            outputs=[y],
            attributes=[Attr("to", AttributeType.INT, int(y_dtype.value))],
            name=tr.fresh_node_name("Cast"),
        )
    else:
        y = rounded

    tr.bind_result_value(y_decl, y)


def _reduce_precision_target_dtype(exponent_bits: int, mantissa_bits: int) -> ir.DataType | None:
    if exponent_bits == 5 and mantissa_bits == 10:
        return ir.DataType.FLOAT16
    if exponent_bits == 8 and mantissa_bits == 7:
        return ir.DataType.BFLOAT16
    if exponent_bits == 8 and mantissa_bits == 23:
        return ir.DataType.FLOAT
    if exponent_bits == 11 and mantissa_bits == 52:
        return ir.DataType.DOUBLE
    return None
