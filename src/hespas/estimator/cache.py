# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import Any
from .result import EstimatorResult
from ..utils.serialisable import Serialisable

@dataclass
class CacheElement(Serialisable):
    """
    This class is a simple container class for an element within the Estimator's
    cache, containing the hardware configuration and the estimation result.
    """
    hw_config: dict[str, Any]
    result: EstimatorResult
    estimator_name: str
    hash: str

    def __post_init__(self):
        if isinstance(self.result, dict):
            self.result = EstimatorResult.from_dict(self.result)
