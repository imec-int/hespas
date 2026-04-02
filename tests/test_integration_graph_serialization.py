# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
"""
Integration tests for dependency graph serialization in the full MLIR parsing workflow.

This module tests the complete workflow from MLIR parsing through splitting,
saving, and loading dependency graphs.
"""

import pytest
import tempfile
import os

from src.hespas.mlir_parser.mlir_splitter import (
    parse_and_split_mlir,
    save_dependency_graph,
    load_dependency_graph,
    validate_dependency_graph,
    linear_split,
    individual_split
)
from src.hespas.mlir_parser.mlir_module import MLIRModule


class TestDependencyGraphIntegration:
    """Integration tests for dependency graph serialization workflow."""

    def setup_method(self):
        """Reset MLIRModule index counter before each test."""
        MLIRModule.reset_index_counter()

    def get_test_mlir_file(self) -> str:
        """Get path to the test MLIR fixture file."""
        # Get the directory of this test file and construct relative path
        test_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(test_dir, "fixtures", "mlir", "test_dependency_graph.mlir")

    def test_parse_split_save_load_workflow(self):
        """Test complete workflow: parse -> split -> save -> load."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Use test MLIR fixture file
            mlir_file = self.get_test_mlir_file()
            output_dir = os.path.join(tmp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            # Parse and split MLIR
            dep_graph = parse_and_split_mlir(mlir_file, output_dir, linear_split)

            # Verify graph was created
            assert dep_graph is not None
            assert len(dep_graph.nodes) > 0

            # Verify pickle file was created during parse_and_split_mlir
            pickle_path = os.path.join(output_dir, "dependency_graph.pkl")
            assert os.path.exists(pickle_path)

            # Load the graph
            loaded_graph = load_dependency_graph(pickle_path)

            # Verify loaded graph
            assert loaded_graph is not None
            assert validate_dependency_graph(loaded_graph)
            assert len(loaded_graph.nodes) == len(dep_graph.nodes)
            assert len(loaded_graph.edges) == len(dep_graph.edges)

            # Verify MLIRModule objects are preserved
            for node_id in dep_graph.nodes:
                original_module = dep_graph.nodes[node_id]['mlir_module']
                loaded_module = loaded_graph.nodes[node_id]['mlir_module']

                assert isinstance(loaded_module, MLIRModule)
                assert loaded_module == original_module
                assert loaded_module.block_type == original_module.block_type
                assert loaded_module.idx == original_module.idx

    def test_save_load_preserves_graph_structure(self):
        """Test that save/load preserves exact graph structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mlir_file = self.get_test_mlir_file()
            output_dir = os.path.join(tmp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            # Parse and split
            original_graph = parse_and_split_mlir(mlir_file, output_dir, linear_split)

            # Load from pickle
            pickle_path = os.path.join(output_dir, "dependency_graph.pkl")
            loaded_graph = load_dependency_graph(pickle_path)

            # Compare graph structures in detail
            assert set(original_graph.nodes) == set(loaded_graph.nodes)
            assert set(original_graph.edges) == set(loaded_graph.edges)

            # Compare node attributes
            for node_id in original_graph.nodes:
                orig_attrs = original_graph.nodes[node_id]
                loaded_attrs = loaded_graph.nodes[node_id]

                # Compare non-mlir_module attributes
                for key in ['color', 'style', 'fillcolor']:
                    if key in orig_attrs:
                        assert orig_attrs[key] == loaded_attrs[key]

                # Compare mlir_module specifically
                assert orig_attrs['mlir_module'] == loaded_attrs['mlir_module']

    def test_mlir_module_index_consistency(self):
        """Test that MLIRModule indices remain consistent across save/load."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mlir_file = self.get_test_mlir_file()
            output_dir = os.path.join(tmp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            # Parse and split
            dep_graph = parse_and_split_mlir(mlir_file, output_dir, linear_split)

            # Record original indices
            original_indices = {}
            for node_id in dep_graph.nodes:
                module = dep_graph.nodes[node_id]['mlir_module']
                original_indices[node_id] = module.idx

            # Record class counter
            original_next_idx = MLIRModule.get_next_idx()

            # Reset counter and load
            MLIRModule.reset_index_counter()
            pickle_path = os.path.join(output_dir, "dependency_graph.pkl")
            loaded_graph = load_dependency_graph(pickle_path)

            # Verify indices are restored
            for node_id in loaded_graph.nodes:
                module = loaded_graph.nodes[node_id]['mlir_module']
                assert module.idx == original_indices[node_id]

            # Verify class counter is restored
            assert MLIRModule.get_next_idx() == original_next_idx

    def test_multiple_save_load_cycles(self):
        """Test multiple save/load cycles preserve data integrity."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mlir_file = self.get_test_mlir_file()
            output_dir = os.path.join(tmp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            # Parse and split
            original_graph = parse_and_split_mlir(mlir_file, output_dir, linear_split)

            current_graph = original_graph

            # Perform multiple save/load cycles
            for cycle in range(3):
                cycle_dir = os.path.join(tmp_dir, f"cycle_{cycle}")
                os.makedirs(cycle_dir, exist_ok=True)

                # Save
                pickle_path = save_dependency_graph(current_graph, cycle_dir)

                # Load
                current_graph = load_dependency_graph(pickle_path)

                # Validate
                assert validate_dependency_graph(current_graph)
                assert len(current_graph.nodes) == len(original_graph.nodes)
                assert len(current_graph.edges) == len(original_graph.edges)

                # Verify MLIRModule objects are still equal
                for node_id in original_graph.nodes:
                    original_module = original_graph.nodes[node_id]['mlir_module']
                    current_module = current_graph.nodes[node_id]['mlir_module']
                    assert original_module == current_module

    def test_save_load_different_split_strategies(self):
        """Test save/load works with different splitting strategies."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mlir_file = self.get_test_mlir_file()

            # Test both linear and data dependency splits
            for split_fn_name, split_fn in [("linear", linear_split), ("individual", individual_split)]:
                output_dir = os.path.join(tmp_dir, f"output_{split_fn_name}")
                os.makedirs(output_dir, exist_ok=True)

                # Reset counter for consistent testing
                MLIRModule.reset_index_counter()

                # Parse and split
                dep_graph = parse_and_split_mlir(mlir_file, output_dir, split_fn)

                # Load from pickle
                pickle_path = os.path.join(output_dir, "dependency_graph.pkl")
                loaded_graph = load_dependency_graph(pickle_path)

                # Validate
                assert validate_dependency_graph(loaded_graph)
                assert len(loaded_graph.nodes) == len(dep_graph.nodes)

                # Verify graph properties match
                for node_id in dep_graph.nodes:
                    orig_module = dep_graph.nodes[node_id]['mlir_module']
                    loaded_module = loaded_graph.nodes[node_id]['mlir_module']
                    assert orig_module == loaded_module

