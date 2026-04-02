# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
"MLIR parser and splitter for HESPAS. Allow top-level analysis of MLIR exports."

from .mlir_splitter import parse_and_split_mlir
from .mlir_parser import MLIRParser
from .mlir_module import MLIRModule

__all__ = ["parse_and_split_mlir", "MLIRParser", "MLIRModule"]
