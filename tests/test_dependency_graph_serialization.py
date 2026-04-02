# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
"""
Unit tests for dependency graph serialization functionality.

This module tests the save/load/validate functions for dependency graphs
containing MLIRModule objects.
"""

import pytest
import tempfile
import networkx as nx
import pickle
import os
from pathlib import Path

from src.hespas.mlir_parser.mlir_splitter import (
    save_dependency_graph,
    load_dependency_graph,
    validate_dependency_graph,
    LoadDependencyException,
    DependencyGraphValidationError
)
from src.hespas.mlir_parser.mlir_module import MLIRModule
from src.hespas.mlir_parser.mlir_common import NodeType


class TestDependencyGraphSerialization:
    """Test cases for dependency graph serialization."""

    def setup_method(self):
        """Reset MLIRModule index counter before each test."""
        MLIRModule.reset_index_counter()

    def create_sample_graph(self) -> nx.DiGraph:
        """Create a sample dependency graph for testing."""
        dep_graph = nx.DiGraph()

        sample_mlir_string_two_op= """module {
  func.func @main(%arg0: tensor<128x128x128xbf16>, %arg1: tensor<128x8x64xbf16>) -> tensor<128x128x8x64xbf16> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128x128xbf16>, tensor<128x8x64xbf16>) -> tensor<128x128x8x64xbf16>
    %1 = stablehlo.tanh %0 : tensor<128x128x8x64xbf16>
    return %1 : tensor<128x128x8x64xbf16>
  }
}"""

        comm_mlir_string="""module {
  func.func @main(%arg0: tensor<1024x10xf32>) -> tensor<1024x10xf32> {
    %0 = "stablehlo.all_reduce"(%arg0) <{replica_groups = dense<0> : tensor<1x1xi64>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1024x10xf32>) -> tensor<1024x10xf32>
    return %0 : tensor<1024x10xf32>
  }
}
"""
        # Create modules
        comp_module = MLIRModule(
            mlir_string=sample_mlir_string_two_op,
            block_type=NodeType.COMP_NODE,
            parent_module="test.mlir"
        )

        comm_module = MLIRModule(
            mlir_string=comm_mlir_string,
            block_type=NodeType.COMM_COLL_NODE,
            parent_module="test.mlir"
        )

        # Add nodes with proper attributes
        dep_graph.add_node(0, color="blue", style="filled", fillcolor="lightblue", mlir_module=comp_module)
        dep_graph.add_node(1, color="red", style="filled", fillcolor="lightcoral", mlir_module=comm_module)

        # Add edge
        dep_graph.add_edge(0, 1)

        return dep_graph

    def test_save_dependency_graph(self):
        """Test saving a dependency graph to pickle file."""
        dep_graph = self.create_sample_graph()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save graph
            pickle_path = save_dependency_graph(dep_graph, tmp_dir)

            # Verify file was created
            assert os.path.exists(pickle_path)
            assert pickle_path.endswith("dependency_graph.pkl")

            # Verify file contains valid pickle data
            with open(pickle_path, 'rb') as f:
                loaded_graph = pickle.load(f)

            assert isinstance(loaded_graph, nx.DiGraph)
            assert len(loaded_graph.nodes) == 2
            assert len(loaded_graph.edges) == 1

    def test_load_dependency_graph(self):
        """Test loading a dependency graph from pickle file."""
        dep_graph = self.create_sample_graph()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save and then load graph
            pickle_path = save_dependency_graph(dep_graph, tmp_dir)
            loaded_graph = load_dependency_graph(pickle_path)

            # Verify structure
            assert len(loaded_graph.nodes) == len(dep_graph.nodes)
            assert len(loaded_graph.edges) == len(dep_graph.edges)

            # Verify MLIRModule objects
            for node_id in dep_graph.nodes:
                original_module = dep_graph.nodes[node_id]['mlir_module']
                loaded_module = loaded_graph.nodes[node_id]['mlir_module']

                assert isinstance(loaded_module, MLIRModule)
                assert loaded_module.mlir_string == original_module.mlir_string
                assert loaded_module.block_type == original_module.block_type
                assert loaded_module.idx == original_module.idx

    def test_load_nonexistent_file(self):
        """Test loading from non-existent pickle file."""
        with pytest.raises(LoadDependencyException, match="FileNotFoundError"):
            load_dependency_graph("/non/existent/path.pkl")

    def test_load_corrupted_pickle(self):
        """Test loading from corrupted pickle file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            # Write invalid pickle data
            tmp.write(b"This is not valid pickle data")
            tmp.flush()

            try:
                with pytest.raises(LoadDependencyException, match="PickleError"):
                    load_dependency_graph(tmp.name)
            finally:
                os.unlink(tmp.name)

    def test_load_wrong_object_type(self):
        """Test loading pickle file with wrong object type."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            # Pickle a list instead of DiGraph
            with open(tmp.name, 'wb') as f:
                pickle.dump([1, 2, 3], f)

            try:
                with pytest.raises(LoadDependencyException, match="Expected nx.DiGraph"):
                    load_dependency_graph(tmp.name)
            finally:
                os.unlink(tmp.name)

    def test_validate_dependency_graph_valid(self):
        """Test validation of a valid dependency graph."""
        dep_graph = self.create_sample_graph()
        assert validate_dependency_graph(dep_graph) is True

    def test_validate_dependency_graph_invalid_type(self):
        """Test validation with invalid graph type."""
        # Use undirected graph instead of directed
        graph = nx.Graph()
        with pytest.raises(DependencyGraphValidationError):
            validate_dependency_graph(graph)

    def test_validate_dependency_graph_cyclic(self):
        """Test validation with cyclic graph."""
        dep_graph = nx.DiGraph()

        # Create modules
        module1 = MLIRModule(mlir_string="""
        module {
          func.func @main() -> () {
            return
          }
        }
        """, block_type=NodeType.COMP_NODE)
        module2 = MLIRModule(mlir_string="""
        module {
          func.func @main() -> () {
            return
          }
        }
        """, block_type=NodeType.COMP_NODE)

        # Add nodes
        dep_graph.add_node(0, mlir_module=module1)
        dep_graph.add_node(1, mlir_module=module2)

        # Add cycle
        dep_graph.add_edge(0, 1)
        dep_graph.add_edge(1, 0)

        with pytest.raises(DependencyGraphValidationError):
            validate_dependency_graph(dep_graph)

    def test_validate_dependency_graph_missing_mlir_module(self):
        """Test validation with missing mlir_module attribute."""
        dep_graph = nx.DiGraph()
        dep_graph.add_node(0, color="blue")  # Missing mlir_module

        with pytest.raises(DependencyGraphValidationError):
            validate_dependency_graph(dep_graph)

    def test_validate_dependency_graph_wrong_mlir_module_type(self):
        """Test validation with wrong mlir_module type."""
        dep_graph = nx.DiGraph()
        dep_graph.add_node(0, mlir_module="not_an_mlir_module")

        with pytest.raises(DependencyGraphValidationError):
            validate_dependency_graph(dep_graph)

    def test_validate_dependency_graph_invalid_block_type(self):
        """Test validation with invalid block type."""
        dep_graph = nx.DiGraph()

        # Create module with valid block type
        module = MLIRModule(mlir_string="""
        module {
          func.func @main() -> () {
            return
          }
        }
        """, block_type=NodeType.COMP_NODE)

        # Monkey patch the block_type property to return invalid value
        module._block_type = "INVALID"

        dep_graph.add_node(0, mlir_module=module)

        with pytest.raises(DependencyGraphValidationError):
            validate_dependency_graph(dep_graph)

    def test_save_load_roundtrip(self):
        """Test complete save/load roundtrip preserves graph integrity."""
        original_graph = self.create_sample_graph()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save graph
            pickle_path = save_dependency_graph(original_graph, tmp_dir)

            # Load graph
            loaded_graph = load_dependency_graph(pickle_path)

            # Validate loaded graph
            assert validate_dependency_graph(loaded_graph) is True

            # Compare structures
            assert len(loaded_graph.nodes) == len(original_graph.nodes)
            assert len(loaded_graph.edges) == len(original_graph.edges)
            assert list(loaded_graph.edges) == list(original_graph.edges)

            # Compare MLIRModule objects
            for node_id in original_graph.nodes:
                orig_module = original_graph.nodes[node_id]['mlir_module']
                loaded_module = loaded_graph.nodes[node_id]['mlir_module']
                assert orig_module == loaded_module

    def test_save_to_custom_output_dir(self):
        """Test saving to custom output directory."""
        dep_graph = self.create_sample_graph()

        with tempfile.TemporaryDirectory() as base_tmp:
            custom_dir = Path(base_tmp) / "custom" / "nested" / "path"

            # Save to nested directory (should create it)
            pickle_path = save_dependency_graph(dep_graph, custom_dir)

            # Verify directory was created and file exists
            assert custom_dir.exists()
            assert os.path.exists(pickle_path)
            assert str(custom_dir / "dependency_graph.pkl") == pickle_path

    def test_save_graph_with_many_nodes(self):
        """Test saving and loading graph with many nodes."""
        dep_graph = nx.DiGraph()

        # Create chain of 10 modules
        modules = []
        for i in range(10):
            module = MLIRModule(
                mlir_string=f"""
        module {{
          func.func @main() -> () {{
            return
          }}
        }}
        """,
                block_type=NodeType.COMP_NODE if i % 2 == 0 else NodeType.COMM_COLL_NODE
            )
            modules.append(module)
            dep_graph.add_node(i, mlir_module=module)

            # Add edge to previous node
            if i > 0:
                dep_graph.add_edge(i-1, i)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save and load
            pickle_path = save_dependency_graph(dep_graph, tmp_dir)
            loaded_graph = load_dependency_graph(pickle_path)

            # Verify all modules are preserved
            assert len(loaded_graph.nodes) == 10
            assert len(loaded_graph.edges) == 9

            for i in range(10):
                loaded_module = loaded_graph.nodes[i]['mlir_module']
                original_module = modules[i]
                assert loaded_module == original_module
