# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import os
import pytest
from src.hespas.chakra_gen.chakra_trace_generator import create_chakra_traces
from src.hespas.chakra_gen.chakra_config import ChakraGenConfig
from src.hespas.mlir_parser.mlir_splitter import parse_and_split_mlir
from src.hespas.mlir_parser.mlir_analyzer import MLIRAnalyzer

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

def test_split():
    """Test MLIR splitter tool.
    This test verifies the MLIR splitter command line tool which splits
    MLIR files into smaller modules based on communication and computation
    operations. See docs/command_line_tools.rst for detailed usage.
    """
    config = ChakraGenConfig(f"{CONFIG_DIR}/configs/config_roofline_a100.json")
    os.makedirs(config.output_dir, exist_ok=True)
    parse_and_split_mlir(config.mlir_file, config.output_dir, config.split_fn, **config.additional_kwargs)

def test_trace_generation():
    """Test Chakra trace generator tool.
    This test verifies the Chakra trace generator which converts MLIR modules
    into execution traces for simulation. See docs/command_line_tools.rst for
    detailed usage.
    """
    create_chakra_traces(f'{CONFIG_DIR}/configs/config_roofline_a100.json', output_dir='output/test_trace_generation')

@pytest.mark.iree
def test_basic_iree_x86_64():
    """Test full HESPAS pipeline with IREE x86_64 backend.
    This test verifies the complete end-to-end pipeline using IREE compiler
    for CPU execution. It includes MLIR splitting, module compilation,
    execution time measurement, and Chakra trace generation.
    See docs/command_line_tools.rst for detailed usage.
    """
    create_chakra_traces(f'{CONFIG_DIR}/configs/config_iree_x64_64.json')

def test_basic_roofline_a100():
    """Test full HESPAS pipeline with Roofline A100 estimation.
    This test verifies the complete pipeline using roofline performance
    modeling for A100 GPU hardware. It estimates execution times without
    requiring actual GPU hardware. See docs/command_line_tools.rst for
    detailed usage.
    """
    create_chakra_traces(f'{CONFIG_DIR}/configs/config_roofline_a100.json')

def test_mlir_analyzer():
    """Test MLIR analyzer tool.
    This test verifies the MLIR analyzer which provides detailed analysis
    of MLIR files including operation statistics, memory usage, and
    performance characteristics. See docs/command_line_tools.rst for
    detailed usage.
    """
    analyzer = MLIRAnalyzer(mlir_path=f'{CONFIG_DIR}/mlir/jax_example.mlir')
    print(analyzer.to_str())

@pytest.mark.xla
def test_xla_estimator():
    """Test full HESPAS pipeline with XLA GPU backend.
    This test verifies the complete end-to-end pipeline using XLA compiler
    for GPU execution. Requires NVIDIA GPU and CUDA support.
    See docs/command_line_tools.rst for detailed usage.
    """
    create_chakra_traces(f'{CONFIG_DIR}/configs/config_xla_gpu_03.json')

