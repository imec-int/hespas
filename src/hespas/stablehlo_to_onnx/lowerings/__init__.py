# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
# lowerings/__init__.py

from .elementwise import register_elementwise_lowerings
from .linear_algebra import register_linear_algebra_lowering
from .convolution import register_convolution_lowering
from .dataflow import register_dataflow_lowerings
from .reduce import register_reduce_lowerings

__all__ = [
    "register_elementwise_lowerings",
    "register_linear_algebra_lowering",
    "register_convolution_lowering",
    "register_dataflow_lowerings",
    "register_reduce_lowerings",
]
