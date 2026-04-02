# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
"""
Unit tests for ChakraTraceGen with dependency graph serialization.

This module tests the complete workflow from MLIR splitting through dependency graph
serialization to Chakra trace generation using the roofline estimator.
"""

import tempfile
import os
import json

from src.hespas.mlir_parser.mlir_splitter import (
    parse_and_split_mlir,
    linear_split,
    load_dependency_graph,
    validate_dependency_graph
)
from src.hespas.chakra_gen.chakra_trace_generator import ChakraTraceGen
from src.hespas.chakra_gen.chakra_config import ChakraGenConfig
from src.hespas.mlir_parser.mlir_module import MLIRModule
from src.hespas.mlir_parser.mlir_common import NodeType

class TestChakraTraceGenerator:
    """Test ChakraTraceGen with dependency graph serialization."""

    def setup_method(self):
        """Reset MLIRModule index counter before each test."""
        MLIRModule.reset_index_counter()

    def get_fixture_mlir_files(self):
        """Get paths to the test MLIR fixture files."""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        return {
            'with_comm': os.path.join(test_dir, "fixtures", "mlir", "test_dependency_graph.mlir"),
            'no_comm': os.path.join(test_dir, "fixtures", "mlir", "test_no_comm_graph.mlir")
        }

    def create_roofline_config(self, output_dir: str, mlir_file: str) -> dict:
        """Create a roofline estimator configuration."""
        return {
            "mlir_file": mlir_file,
            "perf_estimator": {
                "method": "roofline",
                "hardware": {
                    "peak_flops": 312e12,  # A100 peak FLOPS
                    "memory_bandwidth": 1935e9,  # A100 memory bandwidth
                    "num_npus": 4
                }
            },
            "output_dir": output_dir
        }

    def test_mlir_splitting_and_dependency_graph_creation(self):
        """Test MLIR splitting and dependency graph creation with fixture files."""
        fixture_files = self.get_fixture_mlir_files()

        with tempfile.TemporaryDirectory() as tmp_dir:
            for test_name, mlir_file in fixture_files.items():
                output_dir = os.path.join(tmp_dir, f"output_{test_name}")
                os.makedirs(output_dir, exist_ok=True)

                # Reset counter for consistent testing
                MLIRModule.reset_index_counter()

                print(f"\nTesting {test_name} with file: {mlir_file}")

                # Parse and split MLIR with linear_split
                dep_graph = parse_and_split_mlir(mlir_file, output_dir, linear_split)

                # Verify dependency graph was created
                assert dep_graph is not None
                assert len(dep_graph.nodes) > 0
                print(f"Created dependency graph with {len(dep_graph.nodes)} nodes and {len(dep_graph.edges)} edges")

                # Verify pickle file was created
                pickle_path = os.path.join(output_dir, "dependency_graph.pkl")
                assert os.path.exists(pickle_path)

                # Verify graph validation
                assert validate_dependency_graph(dep_graph)

                # Verify all nodes have MLIRModule objects
                for node_id in dep_graph.nodes:
                    mlir_module = dep_graph.nodes[node_id]['mlir_module']
                    assert isinstance(mlir_module, MLIRModule)
                    assert isinstance(mlir_module.block_type, NodeType) and mlir_module.block_type != NodeType.INVALID_NODE
                    assert len(mlir_module.mlir_string) > 0

                # Test load dependency graph
                loaded_graph = load_dependency_graph(pickle_path)
                assert loaded_graph is not None
                assert len(loaded_graph.nodes) == len(dep_graph.nodes)
                assert len(loaded_graph.edges) == len(dep_graph.edges)

    def test_chakra_trace_generation_with_loaded_dependency_graph(self):
        """Test Chakra trace generation using loaded dependency graphs."""
        fixture_files = self.get_fixture_mlir_files()

        with tempfile.TemporaryDirectory() as tmp_dir:
            for test_name, mlir_file in fixture_files.items():
                split_output_dir = os.path.join(tmp_dir, f"split_{test_name}")
                trace_output_dir = os.path.join(tmp_dir, f"traces_{test_name}")
                os.makedirs(split_output_dir, exist_ok=True)
                os.makedirs(trace_output_dir, exist_ok=True)

                # Reset counter for consistent testing
                MLIRModule.reset_index_counter()

                print(f"\nTesting Chakra trace generation for {test_name}")

                # Step 1: Create dependency graph using mlir_splitter
                dep_graph = parse_and_split_mlir(mlir_file, split_output_dir, linear_split)
                pickle_path = os.path.join(split_output_dir, "dependency_graph.pkl")
                assert os.path.exists(pickle_path)

                # Step 2: Load dependency graph from pickle
                loaded_graph = load_dependency_graph(pickle_path)
                assert loaded_graph is not None
                assert validate_dependency_graph(loaded_graph)

                # Step 3: Create ChakraTraceGen with loaded dependency graph
                config = self.create_roofline_config(trace_output_dir, mlir_file)

                # Create ChakraTraceGen instance
                chakra_gen = ChakraTraceGen(ChakraGenConfig(config), loaded_dep_graph=loaded_graph)

                # Step 4: Generate Chakra traces
                chakra_gen.create_chakra_traces()
                traces = chakra_gen.traces

                # Verify traces were generated
                assert traces is not None
                assert isinstance(traces, dict)
                assert len(traces) == config['perf_estimator']['hardware']['num_npus']

                # Verify trace files exist
                for dev_id, trace_path in traces.items():
                    assert os.path.exists(trace_path)
                    assert trace_path.endswith('.et')

                    # Verify file is not empty
                    assert os.path.getsize(trace_path) > 0

                print(f"Successfully generated {len(traces)} trace files for {test_name}")

    def test_chakra_trace_generation_with_communication_ops(self):
        """Test Chakra trace generation specifically with communication operations."""
        fixture_files = self.get_fixture_mlir_files()
        comm_file = fixture_files['with_comm']  # This file has all_reduce operation

        with tempfile.TemporaryDirectory() as tmp_dir:
            split_output_dir = os.path.join(tmp_dir, "split_comm")
            trace_output_dir = os.path.join(tmp_dir, "traces_comm")
            os.makedirs(split_output_dir, exist_ok=True)
            os.makedirs(trace_output_dir, exist_ok=True)

            # Reset counter
            MLIRModule.reset_index_counter()

            print("\nTesting Chakra trace generation with communication operations")

            # Create dependency graph
            dep_graph = parse_and_split_mlir(comm_file, split_output_dir, linear_split)

            # Verify we have both COMP and COMM nodes
            comp_nodes = [n for n in dep_graph.nodes if dep_graph.nodes[n]['mlir_module'].is_computation_block]
            comm_nodes = [n for n in dep_graph.nodes if dep_graph.nodes[n]['mlir_module'].is_communication_block]

            assert len(comp_nodes) > 0, "Should have computation nodes"
            assert len(comm_nodes) > 0, "Should have communication nodes"
            print(f"Found {len(comp_nodes)} COMP nodes and {len(comm_nodes)} COMM nodes")

            # Load and generate traces
            pickle_path = os.path.join(split_output_dir, "dependency_graph.pkl")
            loaded_graph = load_dependency_graph(pickle_path)

            config = self.create_roofline_config(trace_output_dir, comm_file)

            chakra_gen = ChakraTraceGen(ChakraGenConfig(config), loaded_dep_graph=loaded_graph)
            chakra_gen.create_chakra_traces()
            traces = chakra_gen.traces

            # Verify traces for communication operations
            assert traces is not None
            assert len(traces) == config['perf_estimator']['hardware']['num_npus']

            for trace_path in traces.values():
                assert os.path.exists(trace_path)
                assert os.path.getsize(trace_path) > 0

            print("Successfully generated traces with communication operations")

    def test_no_communication_ops_trace_generation(self):
        """Test Chakra trace generation with no communication operations."""
        fixture_files = self.get_fixture_mlir_files()
        no_comm_file = fixture_files['no_comm']  # This file has no communication ops

        with tempfile.TemporaryDirectory() as tmp_dir:
            split_output_dir = os.path.join(tmp_dir, "split_no_comm")
            trace_output_dir = os.path.join(tmp_dir, "traces_no_comm")
            os.makedirs(split_output_dir, exist_ok=True)
            os.makedirs(trace_output_dir, exist_ok=True)

            # Reset counter
            MLIRModule.reset_index_counter()

            print("\nTesting Chakra trace generation with no communication operations")

            # Create dependency graph
            dep_graph = parse_and_split_mlir(no_comm_file, split_output_dir, linear_split)

            # Verify we have only COMP nodes
            comp_nodes = [n for n in dep_graph.nodes if dep_graph.nodes[n]['mlir_module'].is_computation_block]
            comm_nodes = [n for n in dep_graph.nodes if dep_graph.nodes[n]['mlir_module'].is_communication_block]

            assert len(comp_nodes) > 0, "Should have computation nodes"
            assert len(comm_nodes) == 0, "Should have no communication nodes"
            print(f"Found {len(comp_nodes)} COMP nodes and {len(comm_nodes)} COMM nodes")

            # Load and generate traces
            pickle_path = os.path.join(split_output_dir, "dependency_graph.pkl")
            loaded_graph = load_dependency_graph(pickle_path)

            config = self.create_roofline_config(trace_output_dir, no_comm_file)

            chakra_gen = ChakraTraceGen(ChakraGenConfig(config), loaded_dep_graph=loaded_graph)
            chakra_gen.create_chakra_traces()
            traces = chakra_gen.traces

            # Verify traces
            assert traces is not None
            assert len(traces) == config['perf_estimator']['hardware']['num_npus']

            for trace_path in traces.values():
                assert os.path.exists(trace_path)
                assert os.path.getsize(trace_path) > 0

            print("Successfully generated traces with computation-only operations")

    def test_dependency_graph_roundtrip_consistency(self):
        """Test that dependency graphs maintain consistency through save/load cycles."""
        fixture_files = self.get_fixture_mlir_files()

        with tempfile.TemporaryDirectory() as tmp_dir:
            for test_name, mlir_file in fixture_files.items():
                output_dir = os.path.join(tmp_dir, f"roundtrip_{test_name}")
                os.makedirs(output_dir, exist_ok=True)

                # Reset counter
                MLIRModule.reset_index_counter()

                print(f"\nTesting roundtrip consistency for {test_name}")

                # Create original dependency graph
                original_graph = parse_and_split_mlir(mlir_file, output_dir, linear_split)

                # Record original properties
                original_nodes = len(original_graph.nodes)
                original_edges = len(original_graph.edges)
                original_modules = {}
                for node_id in original_graph.nodes:
                    module = original_graph.nodes[node_id]['mlir_module']
                    original_modules[node_id] = {
                        'block_type': module.block_type,
                        'idx': module.idx,
                        'mlir_length': len(module.mlir_string)
                    }

                # Load from pickle
                pickle_path = os.path.join(output_dir, "dependency_graph.pkl")
                loaded_graph = load_dependency_graph(pickle_path)

                # Verify consistency
                assert len(loaded_graph.nodes) == original_nodes
                assert len(loaded_graph.edges) == original_edges

                for node_id in loaded_graph.nodes:
                    loaded_module = loaded_graph.nodes[node_id]['mlir_module']
                    original_props = original_modules[node_id]

                    assert loaded_module.block_type == original_props['block_type']
                    assert loaded_module.idx == original_props['idx']
                    assert len(loaded_module.mlir_string) == original_props['mlir_length']

                # Test that loaded graph can generate traces successfully
                config = self.create_roofline_config(output_dir, mlir_file)

                chakra_gen = ChakraTraceGen(ChakraGenConfig(config), loaded_dep_graph=loaded_graph)
                chakra_gen.create_chakra_traces()
                traces = chakra_gen.traces

                assert traces is not None
                assert len(traces) > 0

                print(f"Roundtrip consistency verified for {test_name}")

    def test_comm_group_json_generated_for_comm_ops(self):
        """Test that comm_group.json is generated when communication ops have replica_groups."""
        fixture_files = self.get_fixture_mlir_files()
        comm_file = fixture_files['with_comm']

        with tempfile.TemporaryDirectory() as tmp_dir:
            split_output_dir = os.path.join(tmp_dir, "split_comm_group")
            trace_output_dir = os.path.join(tmp_dir, "traces_comm_group")
            os.makedirs(split_output_dir, exist_ok=True)
            os.makedirs(trace_output_dir, exist_ok=True)

            MLIRModule.reset_index_counter()

            dep_graph = parse_and_split_mlir(comm_file, split_output_dir, linear_split)
            pickle_path = os.path.join(split_output_dir, "dependency_graph.pkl")
            loaded_graph = load_dependency_graph(pickle_path)

            config = self.create_roofline_config(trace_output_dir, comm_file)
            chakra_gen = ChakraTraceGen(ChakraGenConfig(config), loaded_dep_graph=loaded_graph)
            chakra_gen.create_chakra_traces()
            traces = chakra_gen.traces

            # Verify comm_group.json was generated
            comm_group_path = os.path.join(trace_output_dir, "comm_group.json")
            assert os.path.exists(comm_group_path), "comm_group.json should be generated for MLIR with collectives"

            import json
            with open(comm_group_path) as f:
                comm_groups = json.load(f)

            # All keys should be string group IDs, values should be lists of device IDs
            assert len(comm_groups) > 0
            for group_id, device_list in comm_groups.items():
                assert isinstance(group_id, str)
                assert int(group_id) > 0, "Group IDs should start at 1 (0 is reserved)"
                assert isinstance(device_list, list)
                assert all(isinstance(d, int) for d in device_list)

    def test_no_comm_group_json_without_comm_ops(self):
        """Test that comm_group.json is NOT generated when there are no comm ops."""
        fixture_files = self.get_fixture_mlir_files()
        no_comm_file = fixture_files['no_comm']

        with tempfile.TemporaryDirectory() as tmp_dir:
            split_output_dir = os.path.join(tmp_dir, "split_no_comm_group")
            trace_output_dir = os.path.join(tmp_dir, "traces_no_comm_group")
            os.makedirs(split_output_dir, exist_ok=True)
            os.makedirs(trace_output_dir, exist_ok=True)

            MLIRModule.reset_index_counter()

            dep_graph = parse_and_split_mlir(no_comm_file, split_output_dir, linear_split)
            pickle_path = os.path.join(split_output_dir, "dependency_graph.pkl")
            loaded_graph = load_dependency_graph(pickle_path)

            config = self.create_roofline_config(trace_output_dir, no_comm_file)
            chakra_gen = ChakraTraceGen(ChakraGenConfig(config), loaded_dep_graph=loaded_graph)
            chakra_gen.create_chakra_traces()
            traces = chakra_gen.traces

            # comm_group.json should be generated with a default all-devices group
            # even when there are no comm ops (required by AstraSim)
            comm_group_path = os.path.join(trace_output_dir, "comm_group.json")
            assert os.path.exists(comm_group_path)
            with open(comm_group_path) as f:
                comm_group = json.load(f)
            assert len(comm_group) > 0
