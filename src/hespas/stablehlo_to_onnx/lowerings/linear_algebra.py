# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
from typing import Any, Sequence

import numpy as np
import onnx_ir as ir
from onnx_ir import Attr, AttributeType

from ..registry import LoweringRegistry


def register_linear_algebra_lowering(reg: LoweringRegistry) -> None:
    reg.register("stablehlo.dot_general", lower_dot_general)
    reg.register("stablehlo.dot", lower_dot)
    reg.register("stablehlo.transpose", lower_transpose)
    reg.register("stablehlo.reshape", lower_reshape)


def lower_dot_general(tr: Any, op: Any) -> None:
    """Lower stablehlo.dot_general to ONNX MatMul (+ reshape/transpose)."""
    ins = tr.get_operands(op)
    outs = tr.get_results(op)
    if len(ins) != 2 or len(outs) != 1:
        raise ValueError("stablehlo.dot_general expects 2 inputs -> 1 output")

    lhs, rhs = ins
    out = outs[0]

    dims = getattr(op, "dims", None)
    batch_pairs = dims.get("batch", [])

    lhs_batch = [lb for (lb, _rb) in batch_pairs]
    rhs_batch = [rb for (_lb, rb) in batch_pairs]

    lhs_contract = dims["lhs"] if "lhs" in dims else []
    rhs_contract = dims["rhs"] if "rhs" in dims else []

    lhs_shape = tr.shape_of(lhs)
    rhs_shape = tr.shape_of(rhs)
    if lhs_shape is None or rhs_shape is None:
        raise ValueError("dot_general requires shapes (at least ranks)")

    lhs_rank = len(lhs_shape)
    rhs_rank = len(rhs_shape)

    lhs_batch = [d + lhs_rank if d < 0 else d for d in lhs_batch]
    rhs_batch = [d + rhs_rank if d < 0 else d for d in rhs_batch]
    lhs_contract = [d + lhs_rank if d < 0 else d for d in lhs_contract]
    rhs_contract = [d + rhs_rank if d < 0 else d for d in rhs_contract]

    if len(lhs_batch) != len(rhs_batch):
        raise ValueError("dot_general: batching dims length mismatch")
    if len(lhs_contract) != len(rhs_contract):
        raise ValueError("dot_general: contracting dims length mismatch")

    lhs_batch_set = set(lhs_batch)
    rhs_batch_set = set(rhs_batch)
    lhs_contract_set = set(lhs_contract)
    rhs_contract_set = set(rhs_contract)

    lhs_m_dims = [i for i in range(lhs_rank) if i not in lhs_batch_set | lhs_contract_set]
    rhs_n_dims = [i for i in range(rhs_rank) if i not in rhs_batch_set | rhs_contract_set]

    lhs_perm = lhs_batch + lhs_m_dims + lhs_contract
    rhs_perm = rhs_batch + rhs_contract + rhs_n_dims

    lhs_t = _maybe_transpose(tr, lhs, lhs_perm, name_hint="lhs_t")
    rhs_t = _maybe_transpose(tr, rhs, rhs_perm, name_hint="rhs_t")

    lhs_batch_shape = [lhs_shape[i] for i in lhs_batch]
    rhs_batch_shape = [rhs_shape[i] for i in rhs_batch]
    for a, b in zip(lhs_batch_shape, rhs_batch_shape):
        if isinstance(a, int) and isinstance(b, int) and a != b:
            raise ValueError(f"dot_general: batch dim mismatch {a} vs {b}")

    lhs_m_shape = [lhs_shape[i] for i in lhs_m_dims]
    rhs_n_shape = [rhs_shape[i] for i in rhs_n_dims]
    lhs_k_shape = [lhs_shape[i] for i in lhs_contract]
    rhs_k_shape = [rhs_shape[i] for i in rhs_contract]

    m_flat = _prod(lhs_m_shape)
    n_flat = _prod(rhs_n_shape)
    k_flat_l = _prod(lhs_k_shape)
    k_flat_r = _prod(rhs_k_shape)
    if isinstance(k_flat_l, int) and isinstance(k_flat_r, int) and k_flat_l != k_flat_r:
        raise ValueError(f"dot_general: contracting size mismatch {k_flat_l} vs {k_flat_r}")

    lhs_mm = _reshape(tr, lhs_t, lhs_batch_shape + [m_flat, k_flat_l], name_hint="lhs_mm")
    rhs_mm = _reshape(tr, rhs_t, rhs_batch_shape + [k_flat_r, n_flat], name_hint="rhs_mm")

    mm_out = tr.make_temp(
        dtype=tr.dtype_of(out),
        shape=lhs_batch_shape + [m_flat, n_flat],
        name_hint="matmul_out",
    )
    tr.builder.add_node(
        op_type="MatMul",
        inputs=[lhs_mm, rhs_mm],
        outputs=[mm_out],
        attributes=[],
        name=tr.fresh_node_name("MatMul"),
    )

    out_shape = tr.shape_of(out) or (lhs_batch_shape + lhs_m_shape + rhs_n_shape)
    out_unflat = _reshape(tr, mm_out, out_shape, name_hint="out_unflat")

    tr.bind_result_value(out, out_unflat)


def lower_dot(tr: Any, op: Any) -> None:
    """Lower stablehlo.dot to ONNX MatMul."""
    ins = tr.get_operands(op)
    outs = tr.get_results(op)
    if len(ins) != 2 or len(outs) != 1:
        raise ValueError("stablehlo.dot expects 2 inputs -> 1 output")

    lhs, rhs = ins
    out_decl = outs[0]

    lhs_shape = tr.shape_of(lhs) or list(getattr(op, "lhs_dims", []))
    rhs_shape = tr.shape_of(rhs) or list(getattr(op, "rhs_dims", []))
    if not lhs_shape or not rhs_shape or len(lhs_shape) != 2 or len(rhs_shape) != 2:
        raise ValueError("stablehlo.dot requires rank-2 operands")

    m, k1 = lhs_shape
    k2, n = rhs_shape
    if isinstance(k1, int) and isinstance(k2, int) and k1 != k2:
        raise ValueError(f"stablehlo.dot: K mismatch {k1} vs {k2}")

    out_shape = tr.shape_of(out_decl) or [m, n]

    mm_out = tr.make_temp(
        dtype=tr.dtype_of(out_decl),
        shape=out_shape,
        name_hint="dot_out",
    )
    tr.builder.add_node(
        op_type="MatMul",
        inputs=[lhs, rhs],
        outputs=[mm_out],
        attributes=[],
        name=tr.fresh_node_name("MatMul"),
    )
    tr.bind_result_value(out_decl, mm_out)


def lower_transpose(tr: Any, op: Any) -> None:
    ins = tr.get_operands(op)
    outs = tr.get_results(op)
    if len(ins) != 1 or len(outs) != 1:
        raise ValueError("stablehlo.transpose expects 1 input -> 1 output")

    x = ins[0]
    y_decl = outs[0]

    attrs = getattr(op, "attributes", None)
    perm = attrs.get("permutation", None) if isinstance(attrs, dict) else None

    if not isinstance(perm, (list, tuple)):
        raise ValueError("stablehlo.transpose missing permutation attribute (list[int])")

    perm = [int(p) for p in perm]

    rank = tr.rank_of(x)
    if rank is not None:
        perm = [(p + rank) if p < 0 else p for p in perm]
        if len(perm) != rank:
            raise ValueError(f"stablehlo.transpose: perm length {len(perm)} != rank {rank}")

    if rank is not None and perm == list(range(rank)):
        tr.bind_result_value(y_decl, x)
        return

    x_shape = tr.shape_of(x)
    y_shape = None
    if x_shape is not None and rank is not None:
        y_shape = [x_shape[i] for i in perm]

    y = tr.make_temp(dtype=tr.dtype_of(x), shape=y_shape, name_hint="transpose")
    tr.builder.add_node(
        op_type="Transpose",
        inputs=[x],
        outputs=[y],
        attributes=[_attr_ints("perm", perm)],
        name=tr.fresh_node_name("Transpose"),
    )
    tr.bind_result_value(y_decl, y)


def lower_reshape(tr: Any, op: Any) -> None:
    ins = tr.get_operands(op)
    outs = tr.get_results(op)
    if len(ins) != 1 or len(outs) != 1:
        raise ValueError("stablehlo.reshape expects 1 input -> 1 output")

    x = ins[0]
    y_decl = outs[0]

    new_shape = tr.shape_of(y_decl)
    if new_shape is None:
        explicit = getattr(op, "shape", None)
        if isinstance(explicit, (list, tuple)):
            new_shape = [int(d) for d in explicit]
        else:
            raise ValueError("stablehlo.reshape: output shape is unknown")

    shape_vals = [int(d) if isinstance(d, (int, np.integer)) else -1 for d in new_shape]
    shape_init = tr.builder.const_i64(
        tr.fresh_value_name(),
        shape_vals,
    )

    y = tr.make_temp(dtype=tr.dtype_of(x), shape=list(new_shape), name_hint="reshape")
    tr.builder.add_node(
        op_type="Reshape",
        inputs=[x, shape_init],
        outputs=[y],
        attributes=[],
        name=tr.fresh_node_name("Reshape"),
    )
    tr.bind_result_value(y_decl, y)


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


def _reshape(tr: Any, x: ir.Value, new_shape: Sequence[Any], *, name_hint: str) -> ir.Value:
    y = tr.make_temp(dtype=tr.dtype_of(x), shape=list(new_shape), name_hint=name_hint)
    shape_init = tr.builder.const_i64(
        tr.fresh_value_name(),
        [int(d) if isinstance(d, (int, np.integer)) else -1 for d in new_shape],
    )
    tr.builder.add_node(
        op_type="Reshape",
        inputs=[x, shape_init],
        outputs=[y],
        attributes=[],
        name=tr.fresh_node_name("Reshape"),
    )
    return y


def _prod(dims: Sequence[Any]) -> Any:
    prod = 1
    for d in dims:
        prod *= int(d)
    return int(prod)


def _attr_ints(name: str, vals: Sequence[int]) -> Attr:
    return Attr(name, AttributeType.INTS, list(vals))
