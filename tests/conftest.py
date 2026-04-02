# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import pytest
from src.hespas.estimator.choice import estimator_available, get_conditions

def pytest_addoption(parser):
    parser.addoption("--long-tests", action="store_true", help="Run long unit tests disabled by default")

def pytest_collection_modifyitems(config, items):
    optional_estimator_skips = [
        ("iree", "IREE"),
        ("xla", "XLA"),
        ("cocossim", "Cocossim"),
        ("onnxim", "ONNXim"),
        ("zigzag_onnx", "ZigzagONNX")
    ]
    skips = {
        "slow": (not config.getoption("--long-tests"), "Disabled by default (too long)")
    }
    for estimator_name, display_name in optional_estimator_skips:
        skips[estimator_name] = (not estimator_available(estimator_name) or get_conditions(estimator_name) is not None, "{} not available".format(display_name))

    try:
        import src.hespas.stablehlo_to_onnx.translator
    except ModuleNotFoundError:
        skips["stablehlo_to_onnx"] = (True, "StableHLO to ONNX translator not available")

    for item in items:
        for marker, (do_skip, skip_reason) in skips.items():
            if marker in item.keywords and do_skip:
                item.add_marker(pytest.mark.skip(reason=skip_reason))
                break
