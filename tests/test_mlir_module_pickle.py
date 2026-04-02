"""
Unit tests for MLIRModule pickle support.

This module tests the serialization and deserialization capabilities
of MLIRModule objects using Python's pickle module.
"""
# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT

import pickle
import tempfile
import os

from src.hespas.mlir_parser.mlir_module import MLIRModule
from src.hespas.mlir_parser.mlir_common import NodeType


class TestMLIRModulePickle:
    """Test cases for MLIRModule pickle support."""

    def setup_method(self):
        """Reset MLIRModule index counter before each test."""
        MLIRModule.reset_index_counter()

    def test_pickle_basic_module(self):
        """Test pickling and unpickling a basic MLIRModule."""
        # Create a simple module
        mlir_content = """
        module {
          func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
            return %arg0 : tensor<2x2xf32>
          }
        }
        """

        module = MLIRModule(
            mlir_string=mlir_content,
            block_type=NodeType.COMP_NODE,
            parent_module="test.mlir"
        )

        # Pickle and unpickle
        pickled_data = pickle.dumps(module)
        restored_module = pickle.loads(pickled_data)

        # Verify restoration
        assert restored_module.mlir_string == module.mlir_string
        assert restored_module.block_type == module.block_type
        assert restored_module.parent_module == module.parent_module
        assert restored_module.idx == module.idx
        assert restored_module.metadata == module.metadata

    def test_pickle_comm_module(self):
        """Test pickling a communication module with collective operation."""
        mlir_content = """
        module {
          func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x4xf32> {
            %0 = "stablehlo.all_gather"(%arg0) {
              all_gather_dim = 1 : i64,
              replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
            } : (tensor<2x2xf32>) -> tensor<2x4xf32>
            return %0 : tensor<2x4xf32>
          }
        }
        """

        module = MLIRModule(
            mlir_string=mlir_content,
            block_type=NodeType.COMM_COLL_NODE,
            parent_module="comm_test.mlir"
        )

        # Pickle and unpickle
        pickled_data = pickle.dumps(module)
        restored_module = pickle.loads(pickled_data)

        # Verify restoration
        assert restored_module.collective == "stablehlo.all_gather"
        assert restored_module.is_communication_block
        assert restored_module.is_communication_block

    def test_pickle_multiple_modules_index_preservation(self):
        """Test that index counter is preserved when pickling multiple modules."""
        # Create multiple modules
        modules = []
        for i in range(3):
            module = MLIRModule(
                mlir_string=f"""
        module {{
          func.func @main() -> () {{
            return
          }}
        }}
        """,
                block_type=NodeType.COMP_NODE
            )
            modules.append(module)

        # Verify initial indices
        assert modules[0].idx == 0
        assert modules[1].idx == 1
        assert modules[2].idx == 2
        assert MLIRModule.get_next_idx() == 3

        # Pickle all modules
        pickled_modules = [pickle.dumps(m) for m in modules]

        # Reset counter and restore
        MLIRModule.reset_index_counter()
        restored_modules = [pickle.loads(p) for p in pickled_modules]

        # Verify indices are preserved and counter is restored
        assert restored_modules[0].idx == 0
        assert restored_modules[1].idx == 1
        assert restored_modules[2].idx == 2
        assert MLIRModule.get_next_idx() == 3

    def test_pickle_with_file_path(self):
        """Test pickling module created from file path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as tmp:
            mlir_content = """
            module {
              func.func @main() -> () {
                return
              }
            }
            """
            tmp.write(mlir_content)
            tmp.flush()

            try:
                # Create module from file
                module = MLIRModule(
                    mlir_path=tmp.name,
                    block_type=NodeType.COMP_NODE
                )

                # Pickle and unpickle
                pickled_data = pickle.dumps(module)
                restored_module = pickle.loads(pickled_data)

                # Verify restoration
                assert restored_module.mlir_string == module.mlir_string
                assert restored_module.block_type == module.block_type

            finally:
                os.unlink(tmp.name)

    def test_pickle_with_custom_metadata(self):
        """Test pickling module with custom metadata."""
        module = MLIRModule(
            mlir_string="""
        module {
          func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
            return %arg0 : tensor<2x2xf32>
          }
        }
        """,
            block_type=NodeType.COMP_NODE,
            custom_field="test_value",
            number_field=42
        )

        # Pickle and unpickle
        pickled_data = pickle.dumps(module)
        restored_module = pickle.loads(pickled_data)

        # Verify custom metadata
        assert restored_module.metadata["custom_field"] == "test_value"
        assert restored_module.metadata["number_field"] == 42

    def test_pickle_file_operations(self):
        """Test pickling to and from files."""
        module = MLIRModule(
            mlir_string="""
        module {
          func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
            return %arg0 : tensor<2x2xf32>
          }
        }
        """,
            block_type=NodeType.COMP_NODE,
            parent_module="test.mlir"
        )

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                # Pickle to file
                with open(tmp.name, 'wb') as f:
                    pickle.dump(module, f)

                # Unpickle from file
                with open(tmp.name, 'rb') as f:
                    restored_module = pickle.load(f)

                # Verify restoration
                assert restored_module.mlir_string == module.mlir_string
                assert restored_module.idx == module.idx

            finally:
                os.unlink(tmp.name)

    def test_equality_after_pickle(self):
        """Test that modules are equal after pickle/unpickle."""
        module1 = MLIRModule(
            mlir_string="""
        module {
          func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
            return %arg0 : tensor<2x2xf32>
          }
        }
        """,
            block_type=NodeType.COMP_NODE
        )

        # Pickle and unpickle
        pickled_data = pickle.dumps(module1)
        module2 = pickle.loads(pickled_data)

        # Test equality
        assert module1 == module2
        assert hash(module1) == hash(module2)

    def test_class_methods_after_pickle(self):
        """Test that class methods work properly after pickling."""
        original_next_idx = MLIRModule.get_next_idx()

        module = MLIRModule(
            mlir_string="""
        module {
          func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
            return %arg0 : tensor<2x2xf32>
          }
        }
        """,
            block_type=NodeType.COMP_NODE
        )

        # Pickle and unpickle
        pickled_data = pickle.dumps(module)
        restored_module = pickle.loads(pickled_data)

        # Verify class state
        assert MLIRModule.get_next_idx() == original_next_idx + 1

        # Test setting next idx
        MLIRModule.set_next_idx(100)
        assert MLIRModule.get_next_idx() == 100

        # Create new module
        new_module = MLIRModule(mlir_string="""
        module {
          func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
            return %arg0 : tensor<2x2xf32>
          }
        }
        """, block_type=NodeType.COMP_NODE)
        assert new_module.idx == 100

    def test_mlir_type_preservation_after_pickle(self):
        """Test that MLIR RankedTensorType objects work correctly after pickle/unpickle."""
        mlir_content = """
        module {
          func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<2x2xi32>) -> (tensor<4x4xf32>, tensor<2x2xi32>) {
            return %arg0, %arg1 : tensor<4x4xf32>, tensor<2x2xi32>
          }
        }
        """

        module = MLIRModule(
            mlir_string=mlir_content,
            block_type=NodeType.COMP_NODE
        )

        # Test original MLIR types
        original_inputs = module.input_dims_mlir
        original_outputs = module.output_dims_mlir

        assert len(original_inputs) == 2
        assert len(original_outputs) == 2
        assert original_inputs[0].shape == [4, 4]
        assert original_inputs[1].shape == [2, 2]

        # Pickle and unpickle
        pickled_data = pickle.dumps(module)
        restored_module = pickle.loads(pickled_data)

        # Test reconstructed MLIR types
        restored_inputs = restored_module.input_dims_mlir
        restored_outputs = restored_module.output_dims_mlir

        assert len(restored_inputs) == 2
        assert len(restored_outputs) == 2
        assert restored_inputs[0].shape == [4, 4]
        assert restored_inputs[1].shape == [2, 2]

        # Verify string representations are preserved
        assert module.input_dims == restored_module.input_dims
        assert module.output_dims == restored_module.output_dims

        # Verify MLIR types have same string representation
        for orig, restored in zip(original_inputs, restored_inputs):
            assert str(orig) == str(restored)

        for orig, restored in zip(original_outputs, restored_outputs):
            assert str(orig) == str(restored)
