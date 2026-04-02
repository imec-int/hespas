# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import pytest
import numpy as np

from src.hespas.utils.op_info import OpInfo
from src.hespas.mlir_parser import MLIRParser
from pathlib import Path

@pytest.mark.stablehlo_to_onnx
def test_translate_add_uses_func_return_outputs():
    from src.hespas.stablehlo_to_onnx.translator_api import stablehlo_ops_to_onnx_model

    ops = [
        OpInfo(
            "stablehlo.add",
            operands=["%lhs", "%rhs"],
            results=["%sum"],
            input_types=[([2], "f32"), ([2], "f32")],
            output_types=[([2], "f32")],
        ),
        OpInfo("func.return", input_types=[], output_types=[], operands=["%sum"]),
    ]

    model = stablehlo_ops_to_onnx_model(ops)

    assert len(model.graph.input) == 2
    assert len(model.graph.output) == 1
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "Add"
    assert sorted(list(model.graph.node[0].input)) == sorted([v.name for v in model.graph.input])
    assert list(model.graph.node[0].output) == [model.graph.output[0].name]

@pytest.mark.stablehlo_to_onnx
def test_translate_uses_last_result_as_default_output_without_func_return():
    from src.hespas.stablehlo_to_onnx.translator_api import stablehlo_ops_to_onnx_model

    ops = [
        OpInfo(
            "stablehlo.negate",
            operands=["%arg"],
            results=["%out"],
            input_types=[([4], "f32")],
            output_types=[([4], "f32")],
        )
    ]

    model = stablehlo_ops_to_onnx_model(ops)

    assert len(model.graph.input) == 1
    assert len(model.graph.output) == 1
    assert model.graph.node[0].op_type == "Neg"

@pytest.mark.stablehlo_to_onnx
def test_translate_fallback_without_operands_creates_initializer_output():
    from src.hespas.stablehlo_to_onnx.translator_api import stablehlo_ops_to_onnx_model

    ops = [
        OpInfo(
            "stablehlo.unsupported",
            input_types=[],
            results=["%out"],
            output_types=[([2, 3], "f32")],
        )
    ]

    model = stablehlo_ops_to_onnx_model(ops)

    assert len(model.graph.output) == 1
    assert len(model.graph.node) == 0
    assert len(model.graph.initializer) == 1
    initializer = model.graph.initializer[0]
    assert initializer.name == model.graph.output[0].name
    assert list(initializer.dims) == [2, 3]

@pytest.mark.stablehlo_to_onnx
def test_translate_rejects_unsupported_op_when_fallback_disabled():
    from src.hespas.stablehlo_to_onnx.translator import StableHLOToOnnxTranslator, TranslationOptions

    translator = StableHLOToOnnxTranslator(
        options=TranslationOptions(do_fallback=False)
    )

    with pytest.raises(ValueError, match="Unsupported op: stablehlo.unsupported"):
        translator.translate(
            [
                OpInfo(
                    "stablehlo.unsupported",
                    input_types=[],
                    results=["%out"],
                    output_types=[([1], "f32")],
                )
            ]
        )

@pytest.mark.stablehlo_to_onnx
def test_stablehlo_ops_to_onnx_file_writes_loadable_model(tmp_path):
    import onnx
    from src.hespas.stablehlo_to_onnx.translator_api import stablehlo_ops_to_onnx_file

    ops = [
        OpInfo(
            "stablehlo.add",
            operands=["%lhs", "%rhs"],
            results=["%sum"],
            input_types=[([1], "f32"), ([1], "f32")],
            output_types=[([1], "f32")],
        ),
        OpInfo("func.return", input_types=[], output_types=[], operands=["%sum"]),
    ]
    out_path = tmp_path / "translated.onnx"

    stablehlo_ops_to_onnx_file(ops, str(out_path), model_name="translator_test")

    model = onnx.load(out_path)

    onnx.checker.check_model(model)

    assert model.graph.name == "translator_test"
    assert len(model.graph.output) == 1

@pytest.mark.stablehlo_to_onnx
def test_translate_convolution():
    from src.hespas.stablehlo_to_onnx.translator_api import stablehlo_ops_to_onnx_model

    conv = OpInfo(
        "stablehlo.convolution",
        operands=["%x", "%w"],
        results=["%out"],
        input_types=[([1, 1, 4, 4], "f32"), ([1, 1, 3, 3], "f32")],
        output_types=[([1, 1, 2, 2], "f32")],
    )
    conv.dimension_numbers = {
        "lhs": {
            "batch_dimension": 0,
            "feature_dimension": 1,
            "0_dimension": 2,
            "1_dimension": 3,
        },
        "rhs": {
            "output_feature_dimension": 0,
            "input_feature_dimension": 1,
            "0_dimension": 2,
            "1_dimension": 3,
        },
        "out": {
            "batch_dimension": 0,
            "feature_dimension": 1,
            "0_dimension": 2,
            "1_dimension": 3,
        },
    }
    conv.window = {
        "stride": [1, 1],
        "pad": [[0, 0], [0, 0]],
        "lhs_dilate": [1, 1],
        "rhs_dilate": [1, 1],
    }
    conv.attributes = {"window": conv.window}
    conv.feature_group_count = 1

    model = stablehlo_ops_to_onnx_model(
        [conv, OpInfo("func.return", input_types=[], output_types=[], operands=["%out"])]
    )

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "Conv"

    attrs = getattr(model.graph.node[0], "attribute")
    kernel_shape_attr = [attr for attr in attrs if attr.name == "kernel_shape"][0]
    assert list(kernel_shape_attr.ints) == [3, 3]


@pytest.mark.stablehlo_to_onnx
def test_translate_convolution_lhs_dilation_uses_convtranspose():
    from src.hespas.stablehlo_to_onnx.translator_api import stablehlo_ops_to_onnx_model

    conv = OpInfo(
        "stablehlo.convolution",
        operands=["%x", "%w"],
        results=["%out"],
        input_types=[([1, 1, 4, 4], "f32"), ([1, 1, 3, 3], "f32")],
        output_types=[([1, 1, 6, 6], "f32")],
    )
    conv.dimension_numbers = {
        "lhs": {
            "batch_dimension": 0,
            "feature_dimension": 1,
            "0_dimension": 2,
            "1_dimension": 3,
        },
        "rhs": {
            "output_feature_dimension": 0,
            "input_feature_dimension": 1,
            "0_dimension": 2,
            "1_dimension": 3,
        },
        "out": {
            "batch_dimension": 0,
            "feature_dimension": 1,
            "0_dimension": 2,
            "1_dimension": 3,
        },
    }
    conv.window = {
        "stride": [1, 1],
        "pad": [[2, 1], [2, 1]],
        "lhs_dilate": [2, 2],
        "rhs_dilate": [1, 1],
    }
    conv.attributes = {"window": conv.window}
    conv.feature_group_count = 1

    model = stablehlo_ops_to_onnx_model(
        [conv, OpInfo("func.return", input_types=[], output_types=[], operands=["%out"])]
    )

    conv_nodes = [n for n in model.graph.node if n.op_type == "ConvTranspose"]
    assert len(conv_nodes) == 1

    attrs = getattr(conv_nodes[0], "attribute")
    strides_attr = [attr for attr in attrs if attr.name == "strides"][0]
    assert list(strides_attr.ints) == [2, 2]


@pytest.mark.stablehlo_to_onnx
def test_translate_convolution_rejects_noncanonical_dimension_numbers_shape():
    from src.hespas.stablehlo_to_onnx.translator_api import stablehlo_ops_to_onnx_model

    conv = OpInfo(
        "stablehlo.convolution",
        operands=["%x", "%w"],
        results=["%out"],
        input_types=[([1, 1, 4, 4], "f32"), ([1, 1, 3, 3], "f32")],
        output_types=[([1, 1, 6, 6], "f32")],
    )
    conv.dimension_numbers = {
        "lhs": ["b", "f", "0", "1"],
        "rhs": ["o", "i", "0", "1"],
        "out": ["b", "f", "0", "1"],
    }
    conv.window = {
        "stride": [1, 1],
        "pad": [[2, 1], [2, 1]],
        "lhs_dilate": [2, 2],
        "rhs_dilate": [1, 1],
    }
    conv.attributes = {"window": conv.window}
    conv.feature_group_count = 1

    with pytest.raises(
        ValueError,
        match=r"stablehlo\.convolution expects dimension_numbers\['lhs'\] to be a mapping",
    ):
        stablehlo_ops_to_onnx_model(
            [conv, OpInfo("func.return", input_types=[], output_types=[], operands=["%out"])]
        )


@pytest.mark.stablehlo_to_onnx
def test_translate_constant_rejects_invalid_reshape():
    from src.hespas.stablehlo_to_onnx.translator_api import stablehlo_ops_to_onnx_model

    const = OpInfo(
        "stablehlo.constant",
        input_types=[],
        output_types=[([2, 2], "f32")],
        results=["%out"],
    )
    const.value = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

    with pytest.raises(
        ValueError,
        match=r"stablehlo\.constant: could not reshape constant to declared shape \[2, 2\]",
    ):
        stablehlo_ops_to_onnx_model([const])


@pytest.mark.stablehlo_to_onnx
def test_translate_reduce_window_max_to_maxpool():
    from src.hespas.stablehlo_to_onnx.translator_api import stablehlo_ops_to_onnx_model

    rw = OpInfo(
        "stablehlo.reduce_window",
        operands=["%x", "%init"],
        results=["%out"],
        input_types=[([1, 3, 8, 8], "f32"), ([], "f32")],
        output_types=[([1, 3, 4, 4], "f32")],
    )
    rw.window_dimensions = [1, 1, 3, 3]
    rw.window_strides = [1, 1, 2, 2]
    rw.base_dilations = [1, 1, 1, 1]
    rw.window_dilations = [1, 1, 1, 1]
    rw.padding = [(0, 0), (0, 0), (0, 0), (0, 0)]
    rw.op_line = "%r = stablehlo.reduce_window ... applies stablehlo.maximum ..."

    model = stablehlo_ops_to_onnx_model(
        [rw, OpInfo("func.return", input_types=[], output_types=[], operands=["%out"])]
    )
    assert [n.op_type for n in model.graph.node] == ["MaxPool"]


@pytest.mark.stablehlo_to_onnx
def test_translate_reduce_window_min_to_minpool_pattern():
    from src.hespas.stablehlo_to_onnx.translator_api import stablehlo_ops_to_onnx_model

    rw = OpInfo(
        "stablehlo.reduce_window",
        operands=["%x", "%init"],
        results=["%out"],
        input_types=[([1, 3, 8, 8], "f32"), ([], "f32")],
        output_types=[([1, 3, 4, 4], "f32")],
    )
    rw.window_dimensions = [1, 1, 3, 3]
    rw.window_strides = [1, 1, 2, 2]
    rw.base_dilations = [1, 1, 1, 1]
    rw.window_dilations = [1, 1, 1, 1]
    rw.padding = [(0, 0), (0, 0), (0, 0), (0, 0)]
    rw.op_line = "%r = stablehlo.reduce_window ... applies stablehlo.minimum ..."

    model = stablehlo_ops_to_onnx_model(
        [rw, OpInfo("func.return", input_types=[], output_types=[], operands=["%out"])]
    )
    assert [n.op_type for n in model.graph.node] == ["Neg", "MaxPool", "Neg"]


@pytest.mark.stablehlo_to_onnx
def test_translate_reduce_window_global_add_to_global_average_pool():
    from src.hespas.stablehlo_to_onnx.translator_api import stablehlo_ops_to_onnx_model

    rw = OpInfo(
        "stablehlo.reduce_window",
        operands=["%x", "%init"],
        results=["%out"],
        input_types=[([2, 4, 7, 7], "f32"), ([], "f32")],
        output_types=[([2, 4, 1, 1], "f32")],
    )
    rw.window_dimensions = [1, 1, 7, 7]
    rw.window_strides = [1, 1, 1, 1]
    rw.base_dilations = [1, 1, 1, 1]
    rw.window_dilations = [1, 1, 1, 1]
    rw.padding = [(0, 0), (0, 0), (0, 0), (0, 0)]
    rw.op_line = "%r = stablehlo.reduce_window ... applies stablehlo.add ..."

    model = stablehlo_ops_to_onnx_model(
        [rw, OpInfo("func.return", input_types=[], output_types=[], operands=["%out"])]
    )
    assert [n.op_type for n in model.graph.node] == ["GlobalAveragePool"]


@pytest.mark.stablehlo_to_onnx
def test_translate_reduce_window_nhwc_max_to_transposed_maxpool():
    from src.hespas.stablehlo_to_onnx.translator_api import stablehlo_ops_to_onnx_model

    rw = OpInfo(
        "stablehlo.reduce_window",
        operands=["%x", "%init"],
        results=["%out"],
        input_types=[([256, 112, 112, 64], "bf16"), ([], "bf16")],
        output_types=[([256, 56, 56, 64], "bf16")],
    )
    rw.window_dimensions = [1, 3, 3, 1]
    rw.window_strides = [1, 2, 2, 1]
    rw.base_dilations = [1, 1, 1, 1]
    rw.window_dilations = [1, 1, 1, 1]
    rw.padding = [(0, 0), (0, 1), (0, 1), (0, 0)]
    rw.op_line = "%r = stablehlo.reduce_window ... applies stablehlo.maximum ..."

    model = stablehlo_ops_to_onnx_model(
        [rw, OpInfo("func.return", input_types=[], output_types=[], operands=["%out"])]
    )

    assert [n.op_type for n in model.graph.node] == ["Transpose", "MaxPool", "Transpose"]

@pytest.mark.stablehlo_to_onnx
def test_translate_all_and_check_all_mlir():
    import onnx
    from src.hespas.stablehlo_to_onnx.translator_api import stablehlo_ops_to_onnx_file
    test_dir = Path(__file__).parent
    mlir_dir = test_dir / 'fixtures' / 'stablehlo_to_onnx' / 'mlir'
    onnx_dir = test_dir / 'fixtures' / 'stablehlo_to_onnx' / 'onnx'

    mlir_files = sorted(mlir_dir.glob("*.mlir"))

    if not mlir_files:
        assert False, f"No .mlir files found in {mlir_dir}"
        return

    onnx_dir.mkdir(parents=True, exist_ok=True)

    for mlir_path in mlir_files:
        onnx_path = onnx_dir / f"{mlir_path.stem}.onnx"
        parser = MLIRParser(mlir_path=str(mlir_path))
        ops = parser.ops_list
        stablehlo_ops_to_onnx_file(
            ops,
            str(onnx_path),
            model_name=mlir_path.stem,
            do_fallback=True,
        )
        print(f"Translated {mlir_path.name} -> {onnx_path.name}")

        onnx_model = onnx.load(str(onnx_path))

        try:
            onnx.checker.check_model(onnx_model)
            print(f"ONNX model {onnx_path.name} is valid.")
        except onnx.checker.ValidationError as e:
            assert False, f"ONNX model validation failed for {onnx_path.name}: {e}"
