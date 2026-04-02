# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
"""
Unit tests for parse_replica_groups function in mlir_common.
"""

from src.hespas.mlir_parser.mlir_common import parse_replica_groups


class TestParseReplicaGroups:
    """Test suite for parse_replica_groups."""

    def test_single_group_all_devices(self):
        """Parse a single group containing all 4 devices."""
        mlir = '''%38 = "stablehlo.all_reduce"(%37) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %39 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %39 : tensor<f32>
    }) : (tensor<128x512xf32>) -> tensor<128x512xf32>'''
        assert parse_replica_groups(mlir) == [[0, 1, 2, 3]]

    def test_two_groups_of_two(self):
        """Parse two replica groups of 2 devices each."""
        mlir = '''%5 = "stablehlo.all_reduce"(%4) <{replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %6 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %6 : tensor<f32>
    }) : (tensor<64x128xf32>) -> tensor<64x128xf32>'''
        assert parse_replica_groups(mlir) == [[0, 1], [2, 3]]

    def test_four_groups_of_two(self):
        """Parse four replica groups of 2 devices each (8 devices total)."""
        mlir = '''%10 = "stablehlo.reduce_scatter"(%9) <{replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %11 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %11 : tensor<f32>
    }) : (tensor<256x128xf32>) -> tensor<128x128xf32>'''
        assert parse_replica_groups(mlir) == [[0, 1], [2, 3], [4, 5], [6, 7]]

    def test_scalar_single_device(self):
        """Parse scalar dense<0> format (single device, single group)."""
        mlir = '''%108 = "stablehlo.all_reduce"(%107) <{replica_groups = dense<0> : tensor<1x1xi64>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %109 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %109 : tensor<f32>
    }) : (tensor<784x1024xf32>) -> tensor<784x1024xf32>'''
        assert parse_replica_groups(mlir) == [[0]]

    def test_all_gather(self):
        """Parse replica_groups from an all_gather operation."""
        mlir = '''%7 = "stablehlo.all_gather"(%6) <{all_gather_dim = 0 : i64, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> : (tensor<32x128xf32>) -> tensor<128x128xf32>'''
        assert parse_replica_groups(mlir) == [[0, 1, 2, 3]]

    def test_reduce_scatter(self):
        """Parse replica_groups from a reduce_scatter operation."""
        mlir = '''%5 = "stablehlo.reduce_scatter"(%4) <{replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>, scatter_dimension = 0 : i64}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %6 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %6 : tensor<f32>
    }) : (tensor<256x128xf32>) -> tensor<128x128xf32>'''
        assert parse_replica_groups(mlir) == [[0, 1], [2, 3]]

    def test_all_to_all(self):
        """Parse replica_groups from an all_to_all operation."""
        mlir = '''%3 = "stablehlo.all_to_all"(%2) <{concat_dimension = 1 : i64, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, split_count = 4 : i64, split_dimension = 0 : i64}> : (tensor<128x32xf32>) -> tensor<32x128xf32>'''
        assert parse_replica_groups(mlir) == [[0, 1, 2, 3]]

    def test_collective_permute_returns_none(self):
        """collective_permute uses source_target_pairs, not replica_groups."""
        mlir = '''%2 = "stablehlo.collective_permute"(%1) <{source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>}> : (tensor<128x128xf32>) -> tensor<128x128xf32>'''
        assert parse_replica_groups(mlir) is None

    def test_no_replica_groups_returns_none(self):
        """Non-collective operation returns None."""
        mlir = '''%0 = stablehlo.add %arg0, %arg1 : tensor<128x128xf32>'''
        assert parse_replica_groups(mlir) is None

    def test_non_contiguous_device_ids(self):
        """Parse groups with non-contiguous device IDs."""
        mlir = '''%5 = "stablehlo.all_reduce"(%4) <{replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %6 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %6 : tensor<f32>
    }) : (tensor<64x128xf32>) -> tensor<64x128xf32>'''
        assert parse_replica_groups(mlir) == [[0, 2], [1, 3]]
