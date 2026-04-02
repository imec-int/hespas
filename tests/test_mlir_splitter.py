# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import os
import tempfile
from src.hespas.mlir_parser.mlir_splitter import parse_and_split_mlir, linear_split

MLIR_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "mlir")
NO_COMM_FILE = os.path.join(MLIR_DIR, "test_no_comm_graph.mlir")
COMM_FILE = os.path.join(MLIR_DIR, "test_dependency_graph.mlir")

def run_splitter_test(mlir_file, split_fn, expected_module_count, **kwargs):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a dummy config dict
        config = {
            'mlir_file': mlir_file,
            'output_dir': tmpdir
        }
        modules_dep_graph = parse_and_split_mlir(
            config['mlir_file'],
            config['output_dir'],
            split_fn,
            **kwargs
        )
        # Check that at least one module is produced
        assert len(modules_dep_graph.nodes) == expected_module_count, f'Expected {expected_module_count} modules, got {len(modules_dep_graph.nodes)}'
        # Check that output files exist
        print(modules_dep_graph)
        for node in modules_dep_graph.nodes:
            mini_module = modules_dep_graph.nodes[node]['mlir_module']
            mlir_path = mini_module.module_file
            if not mini_module.is_communication_block:
                assert os.path.exists(os.path.join(tmpdir, mlir_path))
        return modules_dep_graph

def test_no_comm_linear_split():
    modules = run_splitter_test(NO_COMM_FILE, linear_split, 1)
    # Should be a single COMP block (no COMM ops)
    assert all(modules.nodes[m]['mlir_module'].is_computation_block for m in modules.nodes)

def test_comm_linear_split():
    modules = run_splitter_test(COMM_FILE, linear_split, 3)
    # Should contain at least one COMM block
    assert any(modules.nodes[m]['mlir_module'].is_communication_block for m in modules.nodes)

