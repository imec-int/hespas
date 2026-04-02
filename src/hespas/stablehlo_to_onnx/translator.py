# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import onnx_ir as ir
import numpy as np
from onnx_ir import Attr, AttributeType

from .dtype_utils import (
    normalize_shape,
    stablehlo_dtype_to_ir,
)

from .registry import LoweringRegistry
from .lowerings import (
    register_elementwise_lowerings,
    register_linear_algebra_lowering,
    register_convolution_lowering,
    register_dataflow_lowerings,
    register_reduce_lowerings,
)

@dataclass
class TranslationOptions:
    """
    Configuration for `StableHLOToOnnxTranslator`.

    Fields:
        opset: ONNX opset version used when building the model.
        enable_double_precision: Whether unknown/fallback dtype handling should prefer
            `f64` over `f32`.
        model_name: Name assigned to the final ONNX model.
        graph_name: Name assigned to the ONNX graph.
        do_fallback: Whether unsupported ops should use the translator's structural
            fallback path instead of failing immediately.
    """
    opset: int = 18
    enable_double_precision: bool = False
    model_name: str = "stablehlo_model"
    graph_name: str = "main_graph"
    do_fallback: bool = True


class StableHLOToOnnxTranslator:
    """
    Translate a list of parsed StableHLO ops into an ONNX-IR model.

    Translator contract
    -------------------
    The translator expects each op to look like a parsed `OpInfo` instance and provide:

    - `op_name`: StableHLO op name
    - `input_types`: list of `(shape, dtype)` input descriptors
    - `output_types`: list of `(shape, dtype)` output descriptors
    - `operands`: SSA ids consumed by the op
    - `results`: SSA ids produced by the op

    SSA wiring is mandatory. Graph connectivity is derived from `operands` and
    `results`, so callers must provide stable SSA names such as `%arg0`, `%1`, and
    `%out`.

    Graph input and output inference
    --------------------------------
    Any operand SSA id that has not yet been produced by a previous op is treated as a
    graph input. Tensor shape and dtype for such inputs are inferred from the current
    op's `input_types`.

    Graph outputs are chosen as follows:

    - If a `func.return` op is present, its operands are the authoritative graph outputs.
    - Otherwise, the translator uses the results of the last lowered non-`func.return`
      op as graph outputs.

    Naming
    ------
    ONNX value names are allocated from one global counter (`onnx_<n>`), independent
    of source SSA ids. Source SSA ids are used only as internal environment keys.

    Fallback behavior
    -----------------
    When `TranslationOptions.do_fallback=True`, unsupported ops are lowered using a
    structural placeholder path rather than failing immediately. This is useful for
    inspection and experimentation, but the resulting ONNX graph is not guaranteed to be
    semantically equivalent to the original StableHLO program.
    """

    def __init__(self, *, options: TranslationOptions) -> None:
        self.options = options

        # Reuse IRBuilder from jax2onnx.
        from jax2onnx.converter.ir_builder import IRBuilder

        self.builder = IRBuilder(
            opset=options.opset,
            enable_double_precision=options.enable_double_precision,
        )
        self.builder.graph.name = options.graph_name

        self._reg = LoweringRegistry()
        register_elementwise_lowerings(self._reg)
        register_linear_algebra_lowering(self._reg)
        register_convolution_lowering(self._reg)
        register_dataflow_lowerings(self._reg)
        register_reduce_lowerings(self._reg)

        # id(str) -> ir.Value
        self._env: Dict[str, ir.Value] = {}

        self._name_counter: int = 0

    def translate(self, ops: Sequence[Any]) -> ir.Model:
        found_func_return = False
        for idx, op in enumerate(ops):
            op_name = str(getattr(op, "op_name")).strip()
            if op_name == "func.return":
                self._set_outputs_from_func_return(op, idx)
                found_func_return = True
                break

            self._lower_op(op, idx)

        if not found_func_return:
            self._populate_default_outputs(ops)

        # model = ir.Model(
        #     graph=self.builder.graph,
        #     ir_version=None,
        #     producer_name="stablehlo_translator",
        #     model_version=1,
        #     domain="",
        #     doc_string="",
        #     metadata_props=[],
        #     opset_imports={"": self.options.opset},
        # )
        # return model
        return self.builder.to_ir_model(name=self.options.model_name)

    def _set_outputs_from_func_return(self, op: Any, idx: int) -> None:
        if not hasattr(op, "operands"):
            raise ValueError(f"func.return at index {idx} missing operands")

        return_operand_ids = [self._norm_ssa(x) for x in (op.operands or [])]
        ret_inputs_info = getattr(op, "input_types", [])

        # func.return is the authoritative graph-output contract.
        self.builder.outputs.clear()
        for i, oid in enumerate(return_operand_ids):
            v = self._env.get(oid)
            if v is None:
                shape, dtype = ret_inputs_info[i] if i < len(ret_inputs_info) else (None, None)
                ir_dtype = stablehlo_dtype_to_ir(
                    dtype,
                    enable_double_precision=self.options.enable_double_precision,
                )
                v = self._declare_value(name=oid, dtype=ir_dtype, shape=shape)
                self._env[oid] = v
                self.builder.inputs.append(v)
            self.builder.outputs.append(v)

    def _populate_default_outputs(self, ops: Sequence[Any]) -> None:
        # Default graph outputs if none were explicitly added:
        # use the last lowered op's results (ignoring func.return).
        if list(self.builder.outputs):
            return

        last_results: Optional[List[str]] = None
        for op in ops:
            if getattr(op, "op_name", None) == "func.return":
                break
            last_results = list(getattr(op, "results", []))

        if not last_results:
            return

        for rid in last_results:
            v = self._env.get(self._norm_ssa(rid))
            if v is not None:
                self.builder.outputs.append(v)


    def _lower_op(self, op: Any, idx: int) -> None:
        op_name = str(getattr(op, "op_name","")).strip()
        if not op_name:
            raise ValueError(f"Op at index {idx} has no op_name")

        if not hasattr(op, "operands") or not hasattr(op, "results"):
            raise ValueError(
                f"Op '{op_name}' at index {idx} missing operands/results. "
                "Translator requires SSA wiring."
            )

        operands = [self._norm_ssa(x) for x in (op.operands or [])]
        results = [self._norm_ssa(x) for x in (op.results or [])]

        self._ensure_operands(op, operands)
        self._ensure_results(op, results)

        lowering = self._reg.get(op_name)
        if lowering is None:
            if not self.options.do_fallback:
                raise ValueError(f"Unsupported op: {op_name}")
            self._lower_fallback(op, op_name)
            return
        lowering(self, op)

    def _lower_fallback(self, op: Any, op_name: str) -> None:
        """
        Fallback lowering for unsupported ops.
        Reuses the first operand via Reshape/Cast when possible.
        If the op has no operands, synthesize an initializer for each output.
        """
        outs = self.get_results(op)
        if not outs:
            return

        operands = self.get_operands(op)
        for out in outs:
            out_shape = self.shape_of(out)
            out_dtype = self.dtype_of(out)
            if not operands:
                self._bind_fallback_initializer(out, out_dtype, out_shape)
                continue

            produced = operands[0]
            src_shape = self.shape_of(produced)

            if (
                out_shape is not None
                and all(isinstance(d, (int, np.integer)) for d in out_shape)
                and list(src_shape) != list(out_shape)
            ):
                shape_init = self.builder.const_i64(
                    self.fresh_value_name(),
                    [int(d) for d in out_shape],
                )
                reshaped = self.make_temp(
                    dtype=self.dtype_of(produced),
                    shape=out_shape,
                    name_hint="fallback_reshape",
                )
                self.builder.add_node(
                    op_type="Reshape",
                    inputs=[produced, shape_init],
                    outputs=[reshaped],
                    attributes=[],
                    name=self.fresh_node_name("Reshape"),
                )
                produced = reshaped

            if self.dtype_of(produced) != out_dtype:
                casted = self.make_temp(
                    dtype=out_dtype,
                    shape=self.shape_of(produced),
                    name_hint="fallback_cast",
                )
                self.builder.add_node(
                    op_type="Cast",
                    inputs=[produced],
                    outputs=[casted],
                    attributes=[Attr("to", AttributeType.INT, int(out_dtype.value))],
                    name=self.fresh_node_name("Cast"),
                )
                produced = casted

            self.builder.add_node(
                op_type="Identity",
                inputs=[produced],
                outputs=[out],
                attributes=[],
                name=self.fresh_node_name("Identity"),
            )

    def _bind_fallback_initializer(
        self,
        out: ir.Value,
        out_dtype: ir.DataType,
        out_shape: Optional[Sequence[Any]],
    ) -> None:


        np_dtype = out_dtype.numpy()
        if out_shape is None or any(not isinstance(d, (int, np.integer)) for d in out_shape):
            arr = np.array(0, dtype=np_dtype)
            shape = ir.Shape(())
        else:
            arr = np.zeros(tuple(int(d) for d in out_shape), dtype=np_dtype)
            shape = ir.Shape(tuple(int(d) for d in out_shape))

        out.shape = shape
        out.const_value = ir.tensor(arr)
        self.builder.initializers.append(out)

    def _ensure_operands(self, op: Any, operands: List[str]) -> None:
        inputs_info = getattr(op, "input_types", [])
        for i, oid in enumerate(operands):
            shape, dtype = inputs_info[i] if i < len(inputs_info) else (None, None)
            ir_dtype = stablehlo_dtype_to_ir(
                dtype,
                enable_double_precision=self.options.enable_double_precision,
            )
            if oid in self._env:
                cur = self._env[oid]
                cur_shape = self.shape_of(cur)
                cur_dtype = self.dtype_of(cur)
                if self._is_operand_binding_compatible(cur_shape, shape, cur_dtype, ir_dtype):
                    continue
                rebound = self._declare_value(
                    name=oid,
                    dtype=ir_dtype,
                    shape=shape,
                )
                self._env[oid] = rebound
                self.builder.inputs.append(rebound)
                continue
            v = self._declare_value(name=oid, dtype=ir_dtype, shape=shape)
            self._env[oid] = v
            # treat unknown operands as graph inputs
            self.builder.inputs.append(v)

    def _is_operand_binding_compatible(
        self,
        cur_shape: Optional[Sequence[Any]],
        expected_shape: Optional[Sequence[Any]],
        cur_dtype: ir.DataType,
        expected_dtype: ir.DataType,
    ) -> bool:
        if cur_dtype != expected_dtype:
            return False
        if expected_shape is None:
            return True
        if cur_shape is None:
            return False
        if len(cur_shape) != len(expected_shape):
            return False
        for a, b in zip(cur_shape, expected_shape):
            if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)) and int(a) != int(b):
                return False
        return True


    def _ensure_results(self, op: Any, results: List[str]) -> None:
        outputs_info = getattr(op, "output_types", [])
        for i, rid in enumerate(results):
            shape, dtype = outputs_info[i] if i < len(outputs_info) else (None, None)
            ir_dtype = stablehlo_dtype_to_ir(
                dtype,
                enable_double_precision=self.options.enable_double_precision,
            )
            self._env[rid] = self._declare_value(name=rid, dtype=ir_dtype, shape=shape)


    def _declare_value(
        self,
        *,
        name: str,
        dtype: ir.DataType,
        shape: Sequence[Any] | None,
    ) -> ir.Value:
        return ir.Value(
            name=self.fresh_value_name(),
            type=ir.TensorType(dtype),
            shape=(normalize_shape(shape) if shape is not None else None),
        )


    def _norm_ssa(self, name: str) -> str:
        return ("onnx_" + name[1:]) if name.startswith("%") else name


    def bind_result_value(self, declared: ir.Value, produced: ir.Value) -> None:
        for k, v in list(self._env.items()):
            if v is declared:
                self._env[k] = produced
                return


    def fresh_value_name(self) -> str:
        self._name_counter += 1
        return f"onnx_{self._name_counter}"


    def fresh_node_name(self, prefix: str) -> str:
        self._name_counter += 1
        return f"{prefix}_{self._name_counter}"


    def make_temp(
        self,
        *,
        dtype: ir.DataType,
        shape: Sequence[Any] | None,
        name_hint: str,
    ) -> ir.Value:
        name = self.fresh_value_name()
        return self._declare_value(name=name, dtype=dtype, shape=shape)


    def make_temp_like(self, exemplar: ir.Value, *, name_hint: str) -> ir.Value:
        return self.make_temp(
            dtype=self.dtype_of(exemplar),
            shape=self.shape_of(exemplar),
            name_hint=name_hint,
        )


    def dtype_of(self, v: ir.Value) -> ir.DataType:
        t = v.type
        if isinstance(t, ir.TensorType):
            return t.dtype
        return ir.DataType.FLOAT


    def shape_of(self, v: ir.Value) -> Optional[Sequence[Any]]:
        shp = v.shape
        return list(shp.dims) if shp is not None else None


    def rank_of(self, v: ir.Value) -> Optional[int]:
        s = self.shape_of(v)
        return None if s is None else len(s)


    def get_operands(self, op: Any) -> List[ir.Value]:
        operands = [self._norm_ssa(x) for x in (op.operands or [])]
        return [self._env[n] for n in operands]


    def get_results(self, op: Any) -> List[ir.Value]:
        results = [self._norm_ssa(x) for x in (op.results or [])]
        return [self._env[n] for n in results]
