# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import re
from typing import Any, List, Mapping, Sequence, Tuple

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnx_ir as ir
from onnx_ir import Attr, AttributeType

from ..registry import LoweringRegistry


def register_convolution_lowering(reg: LoweringRegistry) -> None:
    reg.register("stablehlo.convolution", lower_convolution)


def lower_convolution(tr: Any, op: Any) -> None:
    """Lower stablehlo.convolution to ONNX Conv or ConvTranspose."""
    ins = tr.get_operands(op)
    outs = tr.get_results(op)
    if len(ins) != 2 or len(outs) != 1:
        raise ValueError("stablehlo.convolution expects 2 inputs -> 1 output")

    x, w = ins
    y_decl = outs[0]

    x_shape = tr.shape_of(x)
    w_shape = tr.shape_of(w)
    y_shape = tr.shape_of(y_decl)
    x_decl_shape = _declared_shape(getattr(op, "input_types", []), 0, "input_types")
    w_decl_shape = _declared_shape(getattr(op, "input_types", []), 1, "input_types")
    y_decl_shape = _declared_shape(getattr(op, "output_types", []), 0, "output_types")

    x_shape_for_layout = x_shape if x_shape is not None and len(x_shape) >= 3 else x_decl_shape
    w_shape_for_layout = w_shape if w_shape is not None and len(w_shape) >= 3 else w_decl_shape
    y_shape_for_layout = y_shape if y_shape is not None and len(y_shape) >= 3 else y_decl_shape
    if x_shape_for_layout is None or w_shape_for_layout is None:
        raise ValueError("convolution requires known ranks (shapes) for input and kernel")

    lhs_dn, rhs_dn, out_dn = _validated_dimension_numbers(op)

    lhs_b = lhs_dn["batch_dimension"]
    lhs_c = lhs_dn["feature_dimension"]

    out_b = out_dn["batch_dimension"]
    out_c = out_dn["feature_dimension"]

    rhs_i = rhs_dn["input_feature_dimension"]
    rhs_o = rhs_dn["output_feature_dimension"]

    lhs_spatial = [lhs_dn['0_dimension'], lhs_dn['1_dimension']]
    rhs_spatial = [rhs_dn['0_dimension'], rhs_dn['1_dimension']]
    out_spatial = [out_dn['0_dimension'], out_dn['1_dimension']]

    n_spatial = len(lhs_spatial)

    window = _validated_window(op, n_spatial)

    strides = window["stride"]
    lhs_dilation = window["lhs_dilate"]
    rhs_dilation = window["rhs_dilate"]
    window_reversal = window.get("reverse")

    if window_reversal is not None and any(bool(b) for b in window_reversal):
        raise NotImplementedError(
            "stablehlo.convolution with window_reversal is not yet supported"
        )

    use_transpose = _is_transposed_convolution(lhs_dilation)
    if use_transpose and any(int(s) != 1 for s in strides):
        raise NotImplementedError(
            "convolution with lhs_dilation and window stride != 1 is not representable as a single ONNX ConvTranspose"
        )

    pads = _padding_pairs_to_onnx_pads(window.get("padding", window.get("pad")), n_spatial)
    group = int(getattr(op, "feature_group_count", 1))

    x = _ensure_operand_shape_compatible(tr, x, x_shape_for_layout, name_hint="conv_lhs")
    w = _ensure_operand_shape_compatible(tr, w, w_shape_for_layout, name_hint="conv_rhs")

    x_perm = [lhs_b, lhs_c] + lhs_spatial
    x_ncx = _maybe_transpose(tr, x, x_perm, name_hint="x_ncx")

    if use_transpose:
        w_perm = [rhs_i, rhs_o] + rhs_spatial
        op_type = "ConvTranspose"
        op_strides = lhs_dilation
        w_name_hint = "w_iox"
    else:
        w_perm = [rhs_o, rhs_i] + rhs_spatial
        op_type = "Conv"
        op_strides = strides
        w_name_hint = "w_oix"

    w_layout = _maybe_transpose(tr, w, w_perm, name_hint=w_name_hint)

    y_ncx_shape = None
    if y_shape_for_layout is not None:
        y_ncx_shape = [y_shape_for_layout[out_b], y_shape_for_layout[out_c]] + [y_shape_for_layout[d] for d in out_spatial]

    y_conv = tr.make_temp(dtype=tr.dtype_of(y_decl), shape=y_ncx_shape, name_hint="y_conv")

    conv_attrs = [
        _attr_ints("strides", op_strides),
        _attr_ints("dilations", rhs_dilation),
        _attr_ints("pads", pads),
        _attr_ints("kernel_shape", [int(w_shape_for_layout[d]) for d in rhs_spatial]),
    ]
    conv_attrs.append(Attr('group', AttributeType.INT, int(group)))

    if use_transpose and y_shape_for_layout is not None:
        conv_attrs.append(_attr_ints("output_shape", [int(y_shape_for_layout[d]) for d in out_spatial]))

    tr.builder.add_node(
        op_type=op_type,
        inputs=[x_ncx, w_layout],
        outputs=[y_conv],
        attributes=conv_attrs,
        name=tr.fresh_node_name(op_type),
    )

    out_rank = len(y_shape_for_layout) if y_shape_for_layout is not None else (2 + n_spatial)
    out_perm = [0] * out_rank
    out_perm[out_b] = 0
    out_perm[out_c] = 1
    for i, d in enumerate(out_spatial):
        out_perm[d] = 2 + i

    y_final = _maybe_transpose(tr, y_conv, out_perm, name_hint="y_out")
    tr.bind_result_value(y_decl, y_final)



def _declared_shape(type_list: Any, idx: int, field_name: str) -> list[int]:
    if idx >= len(type_list):
        raise ValueError(
            f"stablehlo.convolution missing declared shape metadata in {field_name}[{idx}]"
        )
    shape_raw = type_list[idx][0]
    return [int(d) for d in shape_raw]


def _validated_dimension_numbers(op: Any) -> tuple[Mapping[str, int], Mapping[str, int], Mapping[str, int]]:
    dn = getattr(op, "dimension_numbers", None)
    if not isinstance(dn, Mapping):
        raise ValueError(
            "stablehlo.convolution expects dimension_numbers in canonical mapping form"
        )

    required_dims = {
        "lhs": ("batch_dimension", "feature_dimension", "0_dimension", "1_dimension"),
        "rhs": ("output_feature_dimension", "input_feature_dimension", "0_dimension", "1_dimension"),
        "out": ("batch_dimension", "feature_dimension", "0_dimension", "1_dimension"),
    }

    validated: dict[str, Mapping[str, int]] = {}
    for section, required_keys in required_dims.items():
        section_dn = dn.get(section)
        if not isinstance(section_dn, Mapping):
            raise ValueError(
                f"stablehlo.convolution expects dimension_numbers['{section}'] to be a mapping"
            )
        missing_keys = [key for key in required_keys if key not in section_dn]
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise ValueError(
                f"stablehlo.convolution dimension_numbers['{section}'] is missing required keys: {missing}"
            )
        validated[section] = section_dn

    return validated["lhs"], validated["rhs"], validated["out"]


def _validated_window(op: Any, n_spatial: int) -> Mapping[str, Sequence[int]]:
    attrs = getattr(op, "attributes", None)
    if not isinstance(attrs, Mapping):
        raise ValueError("stablehlo.convolution expects an attributes mapping")
    window = attrs.get("window")
    if not isinstance(window, Mapping):
        raise ValueError("stablehlo.convolution expects a window attribute mapping")

    required_lists = ("stride", "lhs_dilate", "rhs_dilate")
    for key in required_lists:
        vals = window.get(key)
        if not isinstance(vals, (list, tuple)):
            raise ValueError(
                f"stablehlo.convolution window is missing required '{key}' list[int]"
            )
        if len(vals) != n_spatial:
            raise ValueError(
                f"stablehlo.convolution window '{key}' rank {len(vals)} does not match spatial rank {n_spatial}"
            )

    return window


def _ensure_operand_shape_compatible(tr: Any, val: ir.Value, expected_shape: Sequence[int] | None, *, name_hint: str) -> ir.Value:
    if expected_shape is None:
        return val
    rank = tr.rank_of(val)
    if rank is not None and rank == len(expected_shape):
        return val
    surrogate = tr.make_temp(dtype=tr.dtype_of(val), shape=list(expected_shape), name_hint=name_hint)
    tr.builder.inputs.append(surrogate)
    return surrogate


def _is_transposed_convolution(lhs_dilation: Sequence[int]) -> bool:
    return any(int(v) != 1 for v in lhs_dilation)

def _maybe_transpose(tr: Any, x: ir.Value, perm: Sequence[int], *, name_hint: str) -> ir.Value:
    rank = tr.rank_of(x)
    if rank is None:
        return x
    if list(perm) == list(range(rank)):
        return x
    shp = tr.shape_of(x)
    out_shape = [shp[i] for i in perm] if shp is not None else None
    y = tr.make_temp(dtype=tr.dtype_of(x), shape=out_shape, name_hint=name_hint)
    tr.builder.add_node(
        op_type="Transpose",
        inputs=[x],
        outputs=[y],
        attributes=[_attr_ints("perm", list(map(int, perm)))],
        name=tr.fresh_node_name("Transpose"),
    )
    return y

def _attr_ints(name: str, vals: Sequence[int]) -> Attr:
    return Attr(name, AttributeType.INTS, [int(v) for v in vals])


def _padding_pairs_to_onnx_pads(padding_pairs: Any, n_spatial: int) -> List[int]:
    if padding_pairs is None:
        return [0] * (2 * n_spatial)

    low = [int(p[0]) for p in padding_pairs]
    high = [int(p[1]) for p in padding_pairs]

    return low + high
