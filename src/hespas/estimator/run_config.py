# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import os
import json
from pathlib import Path
from typing import Optional, Any

from ..utils.dict_tools import recurse_merge
from ..utils.num_tools import conv_bool

class EstimatorRunConfig:
    """
    Configuration object that holds the configuration options required for an Estimator run. Generated from a configuration file,
    or directly through arguments.

    :param config_dict: Configuration dictionary for the specified estimator. Optional, and will override the configuration loaded from file
    :param config_file: Path for a configuration file for the run. Optional, and will be overriden by any other specified options.
    :param disable_cache: Disables the cache for the estimator. Useful for debugging. Will override the setting in either the dict or file
    :param cache_dir: Path to the directory to use for the cache. Optional, will override the setting in either the dict or file.
    :param kwargs: Additional override arguments to the estimator. Optional, will override anything else.
    """
    def __init__(self, *, config_dict: Optional[dict] = None, config_file: Optional[str | Path] = None,
                          disable_cache: Optional[bool] = None, cache_dir: Optional[str | Path] = None, **kwargs):
        self.config = {}
        if config_file is not None:
            with open(config_file) as f:
                self.config = json.load(f)
        if config_dict is not None:
            self.config = recurse_merge(self.config, config_dict)

        self.disable_cache = conv_bool(self.config['disable_cache']) if 'disable_cache' in self.config else disable_cache
        self.cache_dir = self.config.get('cache_dir', cache_dir)
        self.hardware_config = self.config.get('perf_estimator', {}).get('hardware', {})
        self.additional_args = kwargs

    def to_hw_config(self) -> dict[str, Any]:
        """
        Returns a hardware configuration dictionary that be directly fed into the Estimator's initialisation call

        :returns: Estimator hardware config dictionary
        """
        ret_dict = {
            "hw_config": self.hardware_config,
            "cache_dir": self.cache_dir,
            **self.additional_args
        }
        if self.disable_cache is not None:
            ret_dict["disable_cache"] = self.disable_cache
        return ret_dict

class EstimatorSingleRunConfig(EstimatorRunConfig):
    """
    Configuration object that holds the configuration options required for a single Estimator run with a single mlir file.
    Generated from a configuration file, or directly through arguments.

    :param config_dict: Configuration dictionary for the specified estimator. Optional, and will override the configuration loaded from file
    :param config_file: Path for a configuration file for the run. Optional, and will be overriden by any other specified options.
    :param mlir_file: Path for the mlir file for the run. Optional, and will be overriden by any other specified options.
    :param disable_cache: Disables the cache for the estimator. Useful for debugging. Will override the setting in either the dict or file
    :param cache_dir: Path to the directory to use for the cache. Optional, will override the setting in either the dict or file.
    :param kwargs: Additional override arguments to the estimator. Optional, will override anything else.
    """
    def __init__(self, *, config_dict: Optional[dict] = None, config_file: Optional[str | Path] = None, mlir_file: Optional[str | Path] = None,
                          disable_cache: Optional[bool] = None, cache_dir: Optional[str | Path] = None, **kwargs):
        super().__init__(config_dict=config_dict, config_file=config_file, disable_cache=disable_cache, cache_dir=cache_dir, **kwargs)
        if mlir_file is not None:
            self.config['mlir_file'] = mlir_file

        if 'mlir_file' not in self.config:
            raise ValueError("mlir_file missing in config - an input MLIR file must be specified")
        self.mlir_file = self.config['mlir_file']

        if not os.path.isfile(self.mlir_file):
            raise FileNotFoundError("MLIR file '{}' does not exist.".format(self.mlir_file))

        if self.cache_dir is None:
            self.cache_dir = os.path.join(os.path.dirname(self.mlir_file, "cache"))
