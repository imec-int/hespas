"""CLI entry point for StableHLO to ONNX translation."""
# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

from ..mlir_parser import MLIRParser
from .translator_api import stablehlo_ops_to_onnx_file
from ..utils.logging import logger_basic_config, get_log_levels


def run_translation(
    mlir_path: str,
    out_path: str,
    *,
    model_name: str = "stablehlo_model",
    do_fallback: bool = True,
) -> None:
    """Parse a StableHLO .mlir file and translate to ONNX."""
    parser = MLIRParser(mlir_path=mlir_path)
    ops = parser.ops_list
    stablehlo_ops_to_onnx_file(
        ops,
        out_path,
        model_name=model_name,
        do_fallback=do_fallback,
    )

def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Translate a StableHLO .mlir file to ONNX format.")
    ap.add_argument("mlir_path", type=str, help="Path to input .mlir file")
    ap.add_argument("out_path", type=str, help="Path to output .onnx file")
    ap.add_argument("--model-name", type=str, default="stablehlo_model", help="Name of the ONNX model")
    ap.add_argument(
        "--no-fallback",
        action="store_false",
        dest="do_fallback",
        help="Disable fallback to initializer/cast/identity when a StableHLO op is unsupported",
    )
    ap.add_argument("--log-path", default=None, type=str, help="Output path for logging")
    ap.add_argument("--log-level", default='info', type=str, choices=get_log_levels(), help="Set log level")

    return ap

def main():
    args = _build_arg_parser().parse_args()
    logger_basic_config(filename=args.log_path, level=args.log_level)
    mlir_file = Path(args.mlir_path)
    if not mlir_file.exists():
        raise FileNotFoundError(f"Input MLIR file not found: {mlir_file}")

    run_translation(
        str(mlir_file),
        args.out_path,
        model_name=args.model_name,
        do_fallback=args.do_fallback,
    )
    print(f"Translated {mlir_file} -> {args.out_path}")

if __name__ == "__main__":
    main()
