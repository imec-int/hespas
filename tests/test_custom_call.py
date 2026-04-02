# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
"""
Test for stablehlo.custom_call with called_computations attribute.

This test verifies that private functions referenced by custom_call operations
(like TopK) are correctly included in minimodules during splitting.
"""
import os
import tempfile
from src.hespas.mlir_parser.mlir_splitter import parse_and_split_mlir, individual_split
from src.hespas.mlir_parser.mlir_module import MLIRModule

MLIR_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "mlir")
CUSTOM_CALL_FILE = os.path.join(MLIR_DIR, "custom_call_topk.mlir")


def test_custom_call_topk_includes_called_computations():
    """
    Test that custom_call operations with called_computations attribute
    have their referenced private functions included in the minimodule.
    """
    MLIRModule.reset_index_counter()

    with tempfile.TemporaryDirectory() as tmpdir:
        modules_dep_graph = parse_and_split_mlir(
            CUSTOM_CALL_FILE,
            tmpdir,
            individual_split
        )

        # Should produce at least one module
        assert len(modules_dep_graph.nodes) >= 1, "Expected at least one module"

        # Find the module containing the custom_call
        found_custom_call_with_comparator = False
        for node in modules_dep_graph.nodes:
            mini_module = modules_dep_graph.nodes[node]['mlir_module']
            mlir_content = mini_module.mlir_string

            # Print for inspection (use pytest -s to see output)
            print(f"\n=== Mini module {node} ===")
            print(mlir_content)

            # Check if this module contains the custom_call
            if 'stablehlo.custom_call' in mlir_content and '@TopK' in mlir_content:
                # Verify the private comparator function is also included
                assert 'topk_comparator' in mlir_content, (
                    "Private function 'topk_comparator' referenced by called_computations "
                    "should be included in the minimodule"
                )
                found_custom_call_with_comparator = True

        assert found_custom_call_with_comparator, (
            "Should find a minimodule containing both custom_call @TopK and its comparator"
        )
