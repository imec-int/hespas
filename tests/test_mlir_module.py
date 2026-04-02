# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT

"""
Unit tests for the MLIRModule class.

This module contains comprehensive tests for the MLIRModule wrapper class,
testing all functionality including creation, serialization, file I/O, and
property access.
"""
# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT

import pytest
import tempfile
import os
from src.hespas.mlir_parser.mlir_module import MLIRModule
from src.hespas.mlir_parser.mlir_common import NodeType


class TestMLIRModule:
    """Test suite for MLIRModule class."""

    def setup_method(self):
        """Reset the index counter before each test."""
        MLIRModule.reset_index_counter()

    @pytest.fixture
    def sample_mlir_string(self):
        """Sample MLIR module string for testing."""
        return """module {
  func.func @main(%arg0: tensor<128x128x128xbf16>, %arg1: tensor<128x8x64xbf16>) -> tensor<128x128x8x64xbf16> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128x128xbf16>, tensor<128x8x64xbf16>) -> tensor<128x128x8x64xbf16>
    return %0 : tensor<128x128x8x64xbf16>
  }
}"""

    @pytest.fixture
    def sample_mlir_string_two_op(self):
        """Sample MLIR module string for testing."""
        return """module {
  func.func @main(%arg0: tensor<128x128x128xbf16>, %arg1: tensor<128x8x64xbf16>) -> tensor<128x128x8x64xbf16> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128x128xbf16>, tensor<128x8x64xbf16>) -> tensor<128x128x8x64xbf16>
    %1 = stablehlo.tanh %0 : tensor<128x128x8x64xbf16>
    return %1 : tensor<128x128x8x64xbf16>
  }
}"""

    @pytest.fixture
    def comm_mlir_string(self):
        """Sample communication MLIR module string for testing."""
        return """module {
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

    def test_basic_initialization(self, sample_mlir_string):
        """Test basic MLIRModule initialization."""
        module = MLIRModule(
            mlir_string=sample_mlir_string,
            block_type=NodeType.COMP_NODE,
            parent_module="test.mlir"
        )

        assert module.mlir_string == sample_mlir_string
        assert module.idx == 0
        assert module.is_computation_block
        assert module.parent_module == "test.mlir"
        assert module.op_count >= 0  # Should be computed automatically
        assert module.module_file == "mini_module_0.mlir"
        assert module.is_computation_block
        assert not module.is_communication_block

    def test_initialization_without_optional_params(self, sample_mlir_string):
        """Test initialization with minimal parameters."""
        module = MLIRModule(mlir_string=sample_mlir_string)

        assert module.mlir_string == sample_mlir_string
        assert module.idx == 0
        assert module.block_type is None
        assert module.parent_module == ""
        assert module.op_count >= 0
        assert module.module_file == "mini_module_0.mlir"

    def test_unique_index_assignment(self, sample_mlir_string):
        """Test that each module gets a unique incrementing index."""
        modules = []
        for i in range(5):
            module = MLIRModule(
                mlir_string=sample_mlir_string,
                block_type=NodeType.COMP_NODE,
                parent_module="test.mlir"
            )
            modules.append(module)

        # Check indices are unique and incrementing
        for i, module in enumerate(modules):
            assert module.idx == i
            assert module.module_file == f"mini_module_{i}.mlir"

    def test_communication_block(self, comm_mlir_string):
        """Test communication block properties."""
        module = MLIRModule(
            mlir_string=comm_mlir_string,
            block_type=NodeType.COMM_COLL_NODE,
            parent_module="test_comm.mlir"
        )

        assert module.is_communication_block
        assert not module.is_computation_block
        assert module.collective == "stablehlo.all_reduce"
        assert module.is_communication_block

    def test_metadata_population(self, sample_mlir_string):
        """Test that metadata is automatically populated."""
        module = MLIRModule(
            mlir_string=sample_mlir_string,
            block_type=NodeType.COMP_NODE,
            parent_module="test.mlir"
        )

        # Check that metadata contains expected fields
        assert "block_type" in module.metadata
        assert "input_dims" in module.metadata
        assert "output_dims" in module.metadata
        assert "module_file" in module.metadata
        assert "op_count" in module.metadata
        assert "parent_module" in module.metadata

        assert module.metadata["block_type"] == NodeType.COMP_NODE
        assert module.metadata["parent_module"] == "test.mlir"
        assert module.metadata["module_file"] == "mini_module_0.mlir"
        assert module.metadata["op_count"] >= 0

    def test_property_access(self, sample_mlir_string):
        """Test property access methods."""
        module = MLIRModule(
            mlir_string=sample_mlir_string,
            block_type=NodeType.COMP_NODE,
            parent_module="test.mlir"
        )

        assert module.is_computation_block
        assert module.op_count >= 0
        assert module.parent_module == "test.mlir"
        assert module.module_file == "mini_module_0.mlir"
        assert isinstance(module.input_dims, list)
        assert isinstance(module.output_dims, list)

    def test_metadata_copy(self, sample_mlir_string):
        """Test getting metadata copy."""
        module = MLIRModule(
            mlir_string=sample_mlir_string,
            block_type=NodeType.COMP_NODE,
            parent_module="test.mlir"
        )

        metadata_copy = module.get_metadata_copy()
        assert isinstance(metadata_copy, dict)
        assert "block_type" in metadata_copy
        assert metadata_copy["block_type"] == NodeType.COMP_NODE

        # Verify it's a copy, not a reference
        metadata_copy["block_type"] = "MODIFIED"
        assert module.is_computation_block  # Original unchanged

    def test_update_metadata(self, sample_mlir_string):
        """Test metadata updating."""
        module = MLIRModule(
            mlir_string=sample_mlir_string,
            block_type=NodeType.COMP_NODE,
            parent_module="test.mlir"
        )

        original_op_count = module.op_count
        module.update_metadata(custom_field="test_value")

        assert module.metadata["custom_field"] == "test_value"
        assert module.op_count == original_op_count  # Other fields unchanged
        assert module.parent_module == "test.mlir"

    def test_save_to_files_default(self, sample_mlir_string):
        """Test saving module to files with default naming."""
        module = MLIRModule(
            mlir_string=sample_mlir_string,
            block_type=NodeType.COMP_NODE,
            parent_module="test.mlir"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            mlir_path, json_path = module.save_to_files(tmpdir)

            # Check files exist with default names
            assert os.path.exists(mlir_path)
            assert os.path.exists(json_path)
            assert "mini_module_0.mlir" in mlir_path
            assert "mini_module_0.json" in json_path

            # Check file contents
            with open(mlir_path, "r") as f:
                saved_mlir = f.read()
            assert saved_mlir == sample_mlir_string

    def test_save_to_files_custom_name(self, sample_mlir_string):
        """Test saving module to files with custom name."""
        module = MLIRModule(
            mlir_string=sample_mlir_string,
            block_type=NodeType.COMP_NODE,
            parent_module="test.mlir"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            mlir_path, json_path = module.save_to_files(tmpdir, "custom_name")

            # Check files exist with custom names
            assert os.path.exists(mlir_path)
            assert os.path.exists(json_path)
            assert "custom_name.mlir" in mlir_path
            assert "custom_name.json" in json_path

    def test_string_representations(self, sample_mlir_string):
        """Test string representation methods."""
        module = MLIRModule(
            mlir_string=sample_mlir_string,
            block_type=NodeType.COMP_NODE,
            parent_module="test.mlir"
        )

        str_repr = str(module)
        assert "idx=0" in str_repr
        assert "COMP" in str_repr
        assert "mini_module_0.mlir" in str_repr

        repr_str = repr(module)
        print(repr_str)
        assert "MLIRModule" in repr_str
        assert "idx=0" in repr_str
        assert "block_type=NodeType.COMP_NODE" in repr_str

    def test_string_representations_comm_block(self, comm_mlir_string):
        """Test string representations for communication blocks."""
        module = MLIRModule(
            mlir_string=comm_mlir_string,
            block_type=NodeType.COMM_COLL_NODE,
            parent_module="test_comm.mlir"
        )

        str_repr = str(module)
        assert "COMM" in str_repr

    def test_len_method(self, sample_mlir_string):
        """Test __len__ method returns op_count."""
        module = MLIRModule(
            mlir_string=sample_mlir_string,
            block_type=NodeType.COMP_NODE,
            parent_module="test.mlir"
        )

        assert len(module) == module.op_count

    def test_equality(self, sample_mlir_string, sample_mlir_string_two_op):
        """Test equality comparison."""
        MLIRModule.reset_index_counter()

        module1 = MLIRModule(
            mlir_string=sample_mlir_string,
            block_type=NodeType.COMP_NODE,
            parent_module="test.mlir"
        )

        MLIRModule.reset_index_counter()
        module2 = MLIRModule(
            mlir_string=sample_mlir_string,
            block_type=NodeType.COMP_NODE,
            parent_module="test.mlir"
        )

        module3 = MLIRModule(
            mlir_string=sample_mlir_string_two_op,
            block_type=NodeType.COMP_NODE,
            parent_module="test.mlir"
        )

        assert module1 == module2  # Same content, same index after reset
        assert module1 != module3  # Different content
        assert module1 != "not a module"

    def test_additional_kwargs(self, sample_mlir_string):
        """Test that additional kwargs are stored in metadata."""
        module = MLIRModule(
            mlir_string=sample_mlir_string,
            block_type=NodeType.COMP_NODE,
            parent_module="test.mlir",
            custom_field="custom_value",
            another_field=123
        )

        assert module.metadata["custom_field"] == "custom_value"
        assert module.metadata["another_field"] == 123

    def test_reset_index_counter(self, sample_mlir_string):
        """Test resetting the index counter."""
        # Create a few modules
        for i in range(3):
            MLIRModule(
                mlir_string=sample_mlir_string,
                block_type=NodeType.COMP_NODE,
                parent_module="test.mlir"
            )

        # Reset counter
        MLIRModule.reset_index_counter()

        # Next module should have index 0
        module = MLIRModule(
            mlir_string=sample_mlir_string,
            block_type=NodeType.COMP_NODE,
            parent_module="test.mlir"
        )
        assert module.idx == 0

    def test_file_path_input(self):
        """Test creating module from file path."""
        # Create a temporary MLIR file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
            mlir_content = """module {
  func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    func.return %arg0 : tensor<4x4xf32>
  }
}"""
            f.write(mlir_content)
            temp_file_path = f.name

        try:
            # Create module from file path
            module = MLIRModule(
                mlir_path=temp_file_path,
                block_type=NodeType.COMP_NODE
            )

            assert module.mlir_string == mlir_content
            assert str(module.module_path) == temp_file_path
            assert module.op_count >= 0

        finally:
            # Clean up
            os.unlink(temp_file_path)

    def test_collective_extraction(self, comm_mlir_string):
        """Test extraction of collective operations from MLIR strings."""
        # Test all_reduce
        module = MLIRModule(mlir_string=comm_mlir_string, block_type=NodeType.COMM_COLL_NODE)
        assert module.collective == "stablehlo.all_reduce"

    def test_replica_groups_single_device(self, comm_mlir_string):
        """Test replica_groups parsing for single-device scalar format."""
        module = MLIRModule(mlir_string=comm_mlir_string, block_type=NodeType.COMM_COLL_NODE)
        assert module.replica_groups == [[0]]

    def test_replica_groups_all_devices(self):
        """Test replica_groups parsing for a single group of all devices."""
        mlir = """module {
  func.func @main(%arg0: tensor<128x512xf32>) -> tensor<128x512xf32> {
    %0 = "stablehlo.all_reduce"(%arg0) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x512xf32>) -> tensor<128x512xf32>
    return %0 : tensor<128x512xf32>
  }
}"""
        module = MLIRModule(mlir_string=mlir, block_type=NodeType.COMM_COLL_NODE)
        assert module.replica_groups == [[0, 1, 2, 3]]

    def test_replica_groups_split_groups(self):
        """Test replica_groups parsing for multiple communication groups."""
        mlir = """module {
  func.func @main(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = "stablehlo.all_reduce"(%arg0) <{replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x128xf32>) -> tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}"""
        module = MLIRModule(mlir_string=mlir, block_type=NodeType.COMM_COLL_NODE)
        assert module.replica_groups == [[0, 1], [2, 3]]

    def test_replica_groups_none_for_comp_block(self, sample_mlir_string):
        """Test that replica_groups is None for computation blocks."""
        module = MLIRModule(mlir_string=sample_mlir_string, block_type=NodeType.COMP_NODE)
        assert module.replica_groups is None

