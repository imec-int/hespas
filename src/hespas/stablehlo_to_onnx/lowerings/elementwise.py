# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
from typing import Any, Callable

import onnx_ir as ir

from ..registry import LoweringRegistry


_BINARY_SIMPLE: dict[str, str] = {
    "stablehlo.add": "Add",
    "stablehlo.subtract": "Sub",
    "stablehlo.multiply": "Mul",
    "stablehlo.divide": "Div",
    "stablehlo.maximum": "Max",
    "stablehlo.minimum": "Min",
    "stablehlo.power": "Pow",
    "stablehlo.and": "And",
    "stablehlo.or": "Or",
    "stablehlo.xor": "Xor",
}

_UNARY_SIMPLE: dict[str, str] = {
    "stablehlo.abs": "Abs",
    "stablehlo.negate": "Neg",
    "stablehlo.exponential": "Exp",
    "stablehlo.log": "Log",
    "stablehlo.sqrt": "Sqrt",
    "stablehlo.tanh": "Tanh",
    "stablehlo.tan": "Tan",
    "stablehlo.sine": "Sin",
    "stablehlo.cosine": "Cos",
    "stablehlo.not": "Not",
}


def register_elementwise_lowerings(reg: LoweringRegistry) -> None:
    for hlo, onnx_op in _BINARY_SIMPLE.items():
        reg.register(hlo, make_simple_binary(onnx_op))

    for hlo, onnx_op in _UNARY_SIMPLE.items():
        reg.register(hlo, make_simple_unary(onnx_op))

    # special-cases
    reg.register("stablehlo.rsqrt", lower_rsqrt)


def _require_arity(op: Any, ins: list[ir.Value], outs: list[ir.Value], *, n_in: int, n_out: int) -> None:
    if len(ins) != n_in or len(outs) != n_out:
        raise ValueError(f"{op.op_name}: expected {n_in} inputs -> {n_out} output(s)")


def make_simple_binary(op_type: str) -> Callable[[Any, Any], None]:
    def _lower(tr: Any, op: Any) -> None:
        ins = tr.get_operands(op)
        outs = tr.get_results(op)
        _require_arity(op, ins, outs, n_in=2, n_out=1)
        tr.builder.add_node(
            op_type=op_type,
            inputs=[ins[0], ins[1]],
            outputs=[outs[0]],
            attributes=[],
            name=tr.fresh_node_name(op_type),
        )

    return _lower


def make_simple_unary(op_type: str) -> Callable[[Any, Any], None]:
    def _lower(tr: Any, op: Any) -> None:
        ins = tr.get_operands(op)
        outs = tr.get_results(op)
        _require_arity(op, ins, outs, n_in=1, n_out=1)
        tr.builder.add_node(
            op_type=op_type,
            inputs=[ins[0]],
            outputs=[outs[0]],
            attributes=[],
            name=tr.fresh_node_name(op_type),
        )

    return _lower


def lower_rsqrt(tr: Any, op: Any) -> None:
    # rsqrt(x) = 1 / sqrt(x)
    ins = tr.get_operands(op)
    outs = tr.get_results(op)
    _require_arity(op, ins, outs, n_in=1, n_out=1)

    sqrt_out = tr.make_temp_like(ins[0], name_hint="sqrt")
    tr.builder.add_node(
        op_type="Sqrt",
        inputs=[ins[0]],
        outputs=[sqrt_out],
        attributes=[],
        name=tr.fresh_node_name("Sqrt"),
    )
    tr.builder.add_node(
        op_type="Reciprocal",
        inputs=[sqrt_out],
        outputs=[outs[0]],
        attributes=[],
        name=tr.fresh_node_name("Reciprocal"),
    )
