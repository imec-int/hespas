# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol


class LoweringFn(Protocol):
    def __call__(self, tr: Any, op: Any) -> None: ...


class LoweringRegistry:
    def __init__(self) -> None:
        self._lowerings: Dict[str, LoweringFn] = {}


    def register(self, stablehlo_op: str, fn: LoweringFn) -> None:
        key = stablehlo_op.strip()
        if not key:
            raise ValueError("stablehlo_op must be a non-empty string")
        self._lowerings[key] = fn


    def get(self, stablehlo_op: str) -> Optional[LoweringFn]:
        return self._lowerings.get(stablehlo_op.strip())


    def keys(self) -> list[str]:
        return sorted(self._lowerings.keys())
