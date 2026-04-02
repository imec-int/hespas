# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ZIGZAG_FIXTURES_DIR = PROJECT_ROOT.resolve() / "tests" / "fixtures" / "zigzag"

from src.hespas.estimator.result import OpResult
from src.hespas.utils.op_info import OpInfo
from src.hespas.estimator.choice import get_estimator

def _supported_cases() -> dict[str, OpInfo]:
    mat_lhs_shape = [512, 1024]
    mat_rhs_shape = [1024, 768]
    mat_out_shape = [512, 768]
    conv_input_shape = [1, 32, 64, 64]
    conv_kernel_shape = [64, 32, 3, 3]
    conv_output_shape = [1, 64, 62, 62]

    dot_general_op = OpInfo(
        "stablehlo.dot_general",
        operands=["%lhs", "%rhs"],
        results=["%out"],
        input_types=[(mat_lhs_shape, "f32"), (mat_rhs_shape, "f32")],
        output_types=[(mat_out_shape, "f32")],
    )
    dot_general_op.dims = {"lhs": [1], "rhs": [0], "batch": []}
    dot_general_op.lhs_dims = mat_lhs_shape
    dot_general_op.rhs_dims = mat_rhs_shape

    convolution_op = OpInfo(
        "stablehlo.convolution",
        operands=["%x", "%w"],
        results=["%out"],
        input_types=[(conv_input_shape, "f32"), (conv_kernel_shape, "f32")],
        output_types=[(conv_output_shape, "f32")],
    )
    convolution_op.dimension_numbers = {
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
    convolution_op.window = {
        "stride": [1, 1],
        "pad": [[0, 0], [0, 0]],
        "lhs_dilate": [1, 1],
        "rhs_dilate": [1, 1],
    }
    convolution_op.attributes = {"window": convolution_op.window}
    convolution_op.feature_group_count = 1

    return {
        "stablehlo.dot_general": dot_general_op,
        "stablehlo.convolution": convolution_op,
    }


SUPPORTED_CASES = _supported_cases()


def _make_estimator(mapping_name: str = str(ZIGZAG_FIXTURES_DIR / "mapping" / "tpuv3_like.yaml")):
    estimator = get_estimator("zigzag_onnx")(
        {
            "roofline": {
                "peak_flops": 12e13,
                "memory_bandwidth": 9e11,
            },
            "accelerator": ZIGZAG_FIXTURES_DIR / "accelerator" / "tpuv3_like.yaml",
            "mapping": mapping_name,
            "disable_cache": True,
            "in_memory_only_cache": True,
        }
    )
    estimator.print_onnx_model = lambda model: None
    estimator.print_onnx_model_initializers = lambda model: None
    return estimator


@pytest.mark.zigzag_onnx
@pytest.mark.parametrize(
    ("op_name", "expected_node_type"),
    [
        ("stablehlo.dot_general", "MatMul"),
        ("stablehlo.convolution", "Conv"),
    ],
)
def test_translate_mlir_to_onnx_model_uses_per_line_supported_operator(
    op_name: str, expected_node_type: str
):
    estimator = _make_estimator(
        "tpuv3_like_conv.yaml" if op_name == "stablehlo.convolution" else str(ZIGZAG_FIXTURES_DIR / "mapping" / "tpuv3_like.yaml")
    )

    model = estimator.translate_mlir_to_onnx_model(SUPPORTED_CASES[op_name])

    node_types = [node.op_type for node in model.graph.node]
    assert expected_node_type in node_types


@pytest.mark.zigzag_onnx
@pytest.mark.parametrize(
    ("op_name", "mapping_name", "expected_node_type"),
    [
        ("stablehlo.dot_general", str(ZIGZAG_FIXTURES_DIR / "mapping" /"tpuv3_like.yaml"), "MatMul"),
        ("stablehlo.convolution", "tpuv3_like_conv.yaml", "Conv"),
    ],
)
def test_handle_op_invokes_zigzag_for_supported_per_line_ops(
    monkeypatch: pytest.MonkeyPatch,
    op_name: str,
    mapping_name: str,
    expected_node_type: str,
):
    estimator = _make_estimator(mapping_name)
    captured: dict[str, object] = {}

    def fake_get_hardware_performance_zigzag(*, workload, accelerator, mapping, dump_folder):
        captured["node_types"] = [node.op_type for node in workload.graph.node]
        captured["accelerator"] = accelerator
        captured["mapping"] = mapping
        captured["dump_folder"] = dump_folder
        return 0.0, 1234.0, {}

    from src.hespas.estimator import zigzag_onnx_estimator as zigzag_estimator_module
    monkeypatch.setattr(
        zigzag_estimator_module,
        "get_hardware_performance_zigzag",
        fake_get_hardware_performance_zigzag,
    )

    result = estimator.handle_op(SUPPORTED_CASES[op_name])

    assert isinstance(result, OpResult)
    assert result.success is True
    assert float(result.runtime_estimate) == pytest.approx(1234.0 / estimator.frequency)
    assert expected_node_type in captured["node_types"]
    assert captured["accelerator"] == estimator.accelerator
    assert captured["mapping"] == estimator.mapping
    assert "output/zigzag_dump/" in captured["dump_folder"]
