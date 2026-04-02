# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import json
import os
import argparse
import networkx as nx
import shutil
import time
import logging
from functools import lru_cache
from collections import OrderedDict

from .chakra_config import ChakraGenConfig
from ..mlir_parser.mlir_splitter import parse_and_split_mlir, load_dependency_graph, LoadDependencyException
os.environ["TEMPORARILY_DISABLE_PROTOBUF_VERSION_CHECK"] = "true" # Hack to force Protobuf to not be as picky
import chakra.src.generator.generator as chakra
from ..utils.logging import get_str_divider, get_log_levels, logger_basic_config

log = logging.getLogger(__name__)

@lru_cache
def get_block_type(block_type):
    return str(block_type).replace("NodeType.","")

class ChakraTraceGen:
    def __init__(self, config, loaded_dep_graph=None):
        """
        Initialize the pipeline with a compute estimator, mlir module dependency graph, and configuration.
        :param compute_estimator: The compute estimator used for execution.
        :param modules: A data dependent DAG of modules to execute.
        :param config: Configuration dictionary for execution parameters.
        """
        self.config = config
        self.output_dir = self.config.output_dir
        self.estimator_config = self.config.estimator_config
        self.hardware_config = self.estimator_config.hardware_config
        self.mlir_file = self.config.mlir_file
        self.split_fn = self.config.split_fn
        self.num_threads = self.config.num_threads
        self.compute_estimator = self.config.estimator(**self.estimator_config.to_hw_config())
        self.num_devices = self.config.num_devices
        self.module_dep_graph = loaded_dep_graph
        if self.module_dep_graph is None:
            self.module_dep_graph = self.get_modules_dep_graph()
        self.execution_times = {}
        self.traces = dict()
        self.default_stats_print_filter = self.compute_estimator.default_stats_filter
        self.default_stats_out_filter = ["per_op", "per_op_runtime_prop", "total_vs_uncached"]

    def collective_to_chakra_collective(self, collective):
      # TODO
      # Chakra supported primitives: ALL_REDUCE / ALL_TO_ALL/ ALL_GATHER / REDUCE_SCATTER /
      # BROADCAST / BARRIER
      # TODO: stablehlo.collective_permute" --> what is the corresponding in astrasim?
      comm_to_chakra = {
        "stablehlo.all_gather": chakra.ALL_GATHER,
        "stablehlo.all_to_all": chakra.ALL_TO_ALL,
        "stablehlo.reduce_scatter": chakra.REDUCE_SCATTER,
        "stablehlo.all_reduce": chakra.ALL_REDUCE,
        "stablehlo.collective_permute": chakra.ALL_TO_ALL # FIXME
      }
      return comm_to_chakra[collective]

    def create_chakra_traces(self, stats_print_filter=None, stats_out_filter=None):
        """
        Generate Chakra traces for executed modules.

        Generates one trace file per device.  When all collectives use a single
        replica group (all devices together), the traces are identical and
        hardlinked.  When any collective has split replica groups
        (e.g. [[0,1],[2,3]]), each device gets its own file with the correct
        pg_name on comm nodes.
        """
        if stats_print_filter is None:
            stats_print_filter = self.default_stats_print_filter
        if stats_out_filter is None:
            stats_out_filter = self.default_stats_out_filter

        log.info(f">>>> {self.module_dep_graph}")

        results = OrderedDict()

        # Build comm group registry: each unique sub-group of devices gets a
        # stable integer ID.  Group ID 0 is reserved by astra-sim (default
        # all-devices), so IDs start at 1.
        comm_group_registry = {}  # tuple(device_ids) -> group_id
        needs_per_device = False
        next_group_id = 1

        for node in nx.topological_sort(self.module_dep_graph):
            module = self.module_dep_graph.nodes[node]['mlir_module']
            predecessors = list(self.module_dep_graph.predecessors(node))
            block_type = get_block_type(module.block_type)
            block_type_value = module.block_type.value

            if module.is_communication_block:
                dev_to_gid = {}
                if module.replica_groups is not None:
                    if needs_per_device is False and len(module.replica_groups) > 1:
                        needs_per_device = True
                    for key in [tuple(x) for x in module.replica_groups]:
                        gid = comm_group_registry.get(key)
                        if gid is None:
                            gid = next_group_id
                            comm_group_registry[key] = gid
                            next_group_id += 1
                        for dev_id in key:
                            dev_to_gid[dev_id] = gid

                results[node] = {
                    "type": "comm",
                    "block_type": block_type,
                    "block_type_value": block_type_value,
                    "predecessors": predecessors,
                    "comm_size_bytes": module.comm_bytes,
                    "comm_type": self.collective_to_chakra_collective(module.collective),
                    "dev_to_gid": dev_to_gid
                }
            elif module.is_computation_block:
                results[node] = {
                    "type": "comp",
                    "block_type": block_type,
                    "block_type_value": block_type_value,
                    "predecessors": predecessors,
                    "runtime_estimate": self.compute_estimator.get_estimate(module).runtime_estimate
                }
            else:
                raise ValueError("Module is neither communication nor computation")

        # Determine which devices need their own trace file.
        # When traces are identical (no split groups) we generate once and hardlink.
        devices_to_generate = range(self.num_devices) if needs_per_device else range(1)

        log.progress(get_str_divider())
        if needs_per_device:
            log.info(f'Generating per-device chakra traces for {self.num_devices} devices (split replica groups detected).')
        else:
            log.info(f'Generating shared chakra trace for {self.num_devices} devices.')

        for dev in devices_to_generate:
            output_filename = os.path.join(self.output_dir, f"dev.{dev}.et")
            node_id_mapping = {}

            # Reset the global node ID counter so every device's trace
            # has consistent node IDs (required by astra-sim).
            chakra.NODE_ID = 0

            with open(output_filename, "wb") as et:
                chakra.encode_message(et, chakra.GlobalMetadata(version="0.0.4"))

                for node, result in results.items():
                    child_node = chakra.get_node(result["block_type"], result["block_type_value"])
                    child_node.attr.append(chakra.ChakraAttr(name="is_cpu_op", bool_val=False))

                    if result["type"] == "comp":
                        runtime = result["runtime_estimate"]
                        if runtime is None:
                            log.warning("Failed to compute runtime setting it to 0.")
                            runtime = 0

                        duration_value = round(float(runtime) * 1e6) if runtime != 0 else 0
                        try:
                            child_node.duration_micros = int(duration_value)
                        except (ValueError, TypeError, OverflowError) as e:
                            log.warning(f"Failed to convert runtime {duration_value} to integer: {e}. Setting to 0.")
                            child_node.duration_micros = 0
                    elif result["type"] == "comm":
                        child_node.attr.extend([chakra.get_comm_type_attr(result["comm_type"]), chakra.ChakraAttr(name="comm_size", int64_val=result["comm_size_bytes"])])
                        if dev in result["dev_to_gid"]:
                            child_node.attr.append(chakra.ChakraAttr(name="pg_name", string_val=str(result["dev_to_gid"][dev])))
                    else:
                        raise ValueError(f"Invalid module encountered: {module}")

                    if len(result["predecessors"]) > 0:
                        child_node.data_deps.extend([node_id_mapping[pred] for pred in result["predecessors"]])

                    chakra.encode_message(et, child_node)
                    node_id_mapping[node] = child_node.id

            self.traces[dev] = output_filename

        # If traces are identical, hardlink device 0's file to all others
        if not needs_per_device:
            base_file = self.traces[0]
            for dev in range(1, self.num_devices):
                dev_filename = os.path.join(self.output_dir, f"dev.{dev}.et")
                if os.path.isfile(dev_filename):
                    os.remove(dev_filename)
                shutil.copy2(base_file, dev_filename)
                self.traces[dev] = dev_filename

        # Write communicator group JSON for astra-sim
        self._write_comm_group_json(comm_group_registry)

        with open(os.path.join(self.output_dir, "module_stats.json"), "w") as f:
            f.write(json.dumps(self.compute_estimator.stats_tree.to_stats_out(disallow_filter=stats_out_filter), indent=4))
        for statistic in [self.compute_estimator.stats_tree.get_member(k) for k in stats_print_filter if k in self.compute_estimator.stats_tree.members]:
            log.results(str(statistic))

        self.compute_estimator.write_cache()
        log.info('Chakra gen done.')

    def _write_comm_group_json(self, comm_group_registry):
        """Write communicator group JSON file for astra-sim.

        Maps each comm_group_id to its list of device IDs.

        Args:
            comm_group_registry: Dict mapping device-id tuple to group_id.
        """
        if not comm_group_registry:
            comm_group_json = {
                "1": list(range(self.num_devices))
            }
        else:
            comm_group_json = {}
            for device_tuple, group_id in comm_group_registry.items():
                comm_group_json[str(group_id)] = list(device_tuple)

        output_path = os.path.join(self.output_dir, "comm_group.json")
        with open(output_path, "w") as f:
            json.dump(comm_group_json, f, indent=2)
        log.info(f"Wrote communicator group definitions to {output_path}")

    def get_modules_dep_graph(self):
        pickle_path = os.path.join(self.output_dir, "dependency_graph.pkl")
        modules_dep_graph = None
        try:
            # TODO not perfect as module split cannot be interupted and restored midway
            modules_dep_graph = load_dependency_graph(pickle_path)
        except LoadDependencyException:
            modules_dep_graph = None

        if modules_dep_graph is None:
            log.info("Dependency graph not loaded. Generating a new one.")
            modules_dep_graph = parse_and_split_mlir(self.mlir_file, self.output_dir, self.split_fn, num_threads=self.num_threads, **self.config.additional_kwargs)

        if modules_dep_graph is None or len(modules_dep_graph.nodes) <= 0:
            raise ValueError(f"No modules found in the dependency graph for {self.config.mlir_file}")

        log.info(f"Parsed {len(modules_dep_graph.nodes)} modules from the MLIR file.")
        return modules_dep_graph

    def __str__(self):
        """
        Pretty print the execution summary.
        """
        return json.dumps(self.traces, indent=4)

def setup_output_dir(output_dir, clean=True):
    if os.path.exists(output_dir):
        if not os.path.isdir(output_dir):
            raise RuntimeError(f"Output directory {output_dir} exists but is not a directory")
        if clean:
            shutil.rmtree(output_dir)
            log.info(f"Output directory deleted: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    log.info(f"Created output directory: {output_dir}")

def create_chakra_traces(config_path, output_dir=None, mlir_file=None, num_threads=-1,
                         clean=False, stats_print_filter=None, stats_out_filter=None, num_npus=None,
                         split_fn=None, block_lim=None, merge=None):
    # Clean output directory including caching
    split_kwargs = {}
    if block_lim is not None:
        split_kwargs["block_lim"] = block_lim
    if merge is not None:
        split_kwargs["merge"] = merge
    config = ChakraGenConfig(config_path, output_dir=output_dir, mlir_file=mlir_file, clean=clean, num_threads=num_threads, num_npus=num_npus, split_choice=split_fn, split_kwargs=split_kwargs)
    setup_output_dir(config.output_dir, clean=config.clean)

    # estimate perf and create the chakra traces
    trace_gen = ChakraTraceGen(config)
    start_time = time.perf_counter()
    trace_gen.create_chakra_traces(stats_print_filter=stats_print_filter, stats_out_filter=stats_out_filter)
    end_time = time.perf_counter()
    log.progress(get_str_divider())
    log.progress(f"Done. Output directory: {str(config.output_dir)}")
    log.results("Took {}s".format(round(end_time - start_time, 3)))

def get_arg_parser():
    parser = argparse.ArgumentParser(description="Generate trace with input, config, and output files.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config", help="Config file path")
    parser.add_argument("--clean", action='store_true', help="Clean output directory before generating traces")
    parser.add_argument("--mlir_file", type=str, help="Override MLIR file in config")
    parser.add_argument("--num_npus", type=int, help="Override the number of npus in config")
    parser.add_argument("--split_fn", type=str, help="Override split function to use")
    parser.add_argument("--block_lim", type=int, help="Override block size limit for linear_split")
    parser.add_argument("--merge", action="store_true", help="Override merge enable in individual split")
    parser.add_argument("--no-merge", action="store_true", help="Override merge enable in individual split")
    parser.add_argument("--output", type=str, help="Override output directory in config")
    parser.add_argument("--threads", default=-1, type=int, help="Amount of threads to use for splitting. 0 means single-threaded, -1 means the default chosen by ThreadPoolExecutor.")
    parser.add_argument("--log-path", default=None, type=str, help="Output path for logging")
    parser.add_argument("--log-level", default='info', type=str, choices=get_log_levels(), help="Set log level")
    return parser

def main(args=None):
    args = get_arg_parser().parse_args(args)
    logger_basic_config(filename=args.log_path, level=args.log_level)
    indv_merge = None if not (args.merge or args.no_merge) else (True if args.merge and not args.no_merge else False)
    create_chakra_traces(args.config, output_dir=args.output, mlir_file=args.mlir_file, num_threads=args.threads,
                         clean=args.clean, num_npus=args.num_npus, split_fn=args.split_fn, block_lim=args.block_lim, merge=indv_merge)

if __name__ == '__main__':
    main()
