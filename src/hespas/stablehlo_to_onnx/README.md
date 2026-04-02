# StableHLO to ONNX translation

**warning**: Work in progress. Supported lowerings are still limited and unsupported operations may require fallback handling.

## Getting started

The translator takes StableHLO operations parsed by `hespas.mlir_parser` and lowers them into an ONNX model.

Basic CLI usage from the repository root:

```bash
uv run python -m hespas.stablehlo_to_onnx <path-to-input.mlir> <path-to-output.onnx>
```

Fallback lowering is enabled by default. To disable it:

```bash
uv run python -m hespas.stablehlo_to_onnx <path-to-input.mlir> <path-to-output.onnx> --no-fallback
```

## Fallback semantics

When fallback is enabled (default), unsupported StableHLO operations do not fail translation immediately.
Instead, the translator emits a best-effort placeholder lowering so the graph can still be exported.

Current fallback behavior:

- If the unsupported op has at least one operand, the translator reuses the first operand.
- If the declared output shape differs, it inserts an ONNX `Reshape`.
- If the declared output dtype differs, it inserts an ONNX `Cast`.
- It then emits an ONNX `Identity` to bind the produced value to the unsupported op's declared result.
- If the unsupported op has no operands, it synthesizes a zero-valued initializer for each declared output.

This fallback is structural, not semantic. It is intended to keep translation moving for inspection,
debugging, or downstream tooling experiments. It does not preserve the real meaning of unsupported
operations and should not be treated as numerically correct model conversion.

Use `--no-fallback` or `do_fallback=False` if you want translation to fail fast on unsupported ops.

## Python usage

Parse an MLIR file and write the translated ONNX model to disk:

```python
from hespas.mlir_parser import MLIRParser
from hespas.stablehlo_to_onnx.translator_api import stablehlo_ops_to_onnx_file

parser = MLIRParser(mlir_path="path/to/model.mlir")
ops = parser.ops_list

stablehlo_ops_to_onnx_file(
    ops,
    "model.onnx",
)
```

If you want the ONNX model object directly:

```python
from hespas.stablehlo_to_onnx.translator_api import stablehlo_ops_to_onnx_model

model = stablehlo_ops_to_onnx_model(ops)
```

## Supported lowerings

Currently supported StableHLO operators are:

- Elementwise binary:
  `stablehlo.add`, `stablehlo.subtract`, `stablehlo.multiply`, `stablehlo.divide`,
  `stablehlo.maximum`, `stablehlo.minimum`, `stablehlo.power`,
  `stablehlo.and`, `stablehlo.or`, `stablehlo.xor`
- Elementwise unary:
  `stablehlo.abs`, `stablehlo.negate`, `stablehlo.exponential`, `stablehlo.log`,
  `stablehlo.sqrt`, `stablehlo.tanh`, `stablehlo.tan`, `stablehlo.sine`,
  `stablehlo.cosine`, `stablehlo.rsqrt`, `stablehlo.not`
- Linear algebra:
  `stablehlo.dot_general`, `stablehlo.dot`, `stablehlo.convolution`,
  `stablehlo.transpose`, `stablehlo.reshape`
- Dataflow-oriented:
  `stablehlo.constant`, `stablehlo.convert`, `stablehlo.broadcast_in_dim`,
  `stablehlo.concatenate`, `stablehlo.select`, `stablehlo.slice`, `stablehlo.compare`
- Reduce:
  `stablehlo.reduce`

These lower roughly to standard ONNX operators such as `Add`, `MatMul`, `Conv`,
`Reshape`, `Concat`, `Slice`, and `Where`.

Current `stablehlo.reduce` behavior:
- Scalar/tensor reduce form (`2 inputs -> 1 output`) maps reducers to:
  - `stablehlo.add` -> `ReduceSum`
  - `stablehlo.maximum` -> `ReduceMax`
  - `stablehlo.minimum` -> `ReduceMin`
  - `stablehlo.and` -> `ReduceMin` (boolean)
  - `stablehlo.or` -> `ReduceMax` (boolean)
- Tuple reduce form (`4 inputs -> 2 outputs`) supports argmin and argmax reducers over a single axis:
  - reducer compare `GT` -> `ArgMax`-based lowering
  - reducer compare `LT` -> `ArgMin`-based lowering
  - selected value/index are materialized with `GatherElements` + `Squeeze` (+ optional `Cast` for index dtype)

Current `stablehlo.convolution` behavior:

- Maps to ONNX `Conv` or `ConvTranspose` based on semantics:
  - `lhs_dilation` all ones -> `Conv` (down-sampling)
  - any `lhs_dilation != 1` -> `ConvTranspose` (up-sampling)
- Uses `dimension_numbers` to normalize input/kernel/output layouts.
- Inserts layout `Transpose` ops as needed so ONNX operators run in channel-first layout and then restores declared output layout.
- Supports `strides`, `rhs_dilation`, `padding`, and `feature_group_count`.

## Extending the translator

Lowerings are registered through `LoweringRegistry` in `translator.py`.

To add a new lowering:

1. Add the lowering implementation in one of the files under `lowerings/`, or create a new lowering module.
2. Register it through the relevant `register_*` helper so it is added to the translator registry.
3. Use translator helpers to read operands and results, create temporary values, and bind translated outputs back to declared results.
