# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import os
import json
from pathlib import Path
from ..estimator.choice import get_estimator
from ..estimator.run_config import EstimatorRunConfig
from ..mlir_parser.mlir_splitter import dispatch_split_fn

class ChakraGenConfig:
    def __init__(self, input, output_dir=None, mlir_file=None, clean=None,
                              disable_cache=None, cache_dir=None, num_threads=-1,
                              num_npus=None, split_choice=None, split_kwargs=None):
        self.config = None
        if isinstance(input, str) or isinstance(input, Path):
            with open(input) as f:
                self.config = json.load(f)
        elif isinstance(input, dict):
            self.config = input
        else:
            raise ValueError("ChakraGenConfig requires either a path as its first argument, or a config dictionary; not '{}'".format(type(input)))

        self.estimator_kwargs = {}
        self.clean = bool(self.config['clean']) if 'clean' in self.config else clean
        self.disable_cache = bool(self.config['disable_cache']) if 'disable_cache' in self.config else disable_cache
        if self.disable_cache is not None:
            self.estimator_kwargs['disable_cache'] = self.disable_cache
        self.num_threads = int(self.config.get('num_threads', num_threads))

        if output_dir is not None:
            self.config['output_dir'] = output_dir
        if mlir_file is not None:
            self.config['mlir_file'] = mlir_file
        if "perf_estimator" not in self.config:
            raise ValueError("perf_estimator field missing in config - a Perf Estimator must be specified")
        estimator_dict = self.config['perf_estimator']

        if num_npus is not None:
            self.num_devices = int(num_npus)
        else:
            self.num_devices = estimator_dict.get('hardware', {}).get('num_npus', 1)

        self.config["perf_estimator"]["hardware"]["num_npus"] = self.num_devices
        if 'output_dir' not in self.config or self.config['output_dir'] is None:
            raise ValueError("output_dir field missing in config - an output directory must be specified")
        self.output_dir = self.config['output_dir']

        if 'mlir_file' not in self.config:
            raise ValueError("mlir_file field missing in config - an input MLIR file must be specified")
        self.mlir_file = self.config['mlir_file']

        if not os.path.isfile(self.mlir_file):
            raise FileNotFoundError("MLIR file '{}' does not exist.".format(self.mlir_file))

        self.cache_dir = self.config.get('cache_dir', os.path.join(self.output_dir, "cache"))
        self.estimator_kwargs['cache_dir'] = self.cache_dir
        if split_choice is None:
            split_choice = self.config.get('split_strategy', 'linear_split').lower()
        if split_choice not in dispatch_split_fn:
            raise ValueError("Split Strategy '{}' does not exist - choose from {}".format(split_choice, list(dispatch_split_fn.keys())))
        self.split_fn = dispatch_split_fn[split_choice]

        self.additional_kwargs = split_kwargs if split_kwargs is not None else {}
        # TODO: Move this into mlir_splitter.py in some sort of sensical way
        if split_choice == "linear_split":
            if "block_lim" not in self.additional_kwargs:
                self.additional_kwargs["block_lim"] = self.config.get("block_lim", 1024)
            if "seperator" not in self.additional_kwargs:
                self.additional_kwargs["separator"] = self.config.get("separator", "is_communication_op")
            if "seperator_policy" not in self.additional_kwargs:
                self.additional_kwargs["separator_policy"] = self.config.get("separator_policy", "isolated").upper()
            if "seperator_node_type" not in self.additional_kwargs:
                self.additional_kwargs["separator_node_type"] = self.config.get("separator_node_type", "comm_coll_node").upper()
        elif split_choice == "individual_split":
            if "merge" not in self.additional_kwargs:
                self.additional_kwargs["merge"] = self.config.get("merge", False)

        if 'method' not in estimator_dict:
            raise ValueError("An estimator must be specified in the 'method' key of the 'perf_estimator' object")
        estimator_name = estimator_dict['method'].lower()
        self.estimator = get_estimator(estimator_name)
        self.estimator_config = EstimatorRunConfig(config_dict={'perf_estimator': {'hardware': estimator_dict.get('hardware', {})}}, **self.estimator_kwargs)
