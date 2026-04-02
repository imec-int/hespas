# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import os
import json
import pickle
import time
from pathlib import Path
from typing import Optional, Union
import concurrent.futures
from .mlir_analyzer import MLIRAnalyzer
import logging
from tqdm import tqdm
import networkx as nx
import psutil
from enum import Enum, auto
import argparse
import sys

from .mlir_common import (
    is_communication_op,
    is_return_op,
    create_new_module_with_operations,
    NodeType,
    store_private_functions,
    op_classifiers
)
from .mlir_module import MLIRModule
from ..utils.logging import get_str_divider, get_log_levels, logger_basic_config

log = logging.getLogger(__name__)

class LoadDependencyException(Exception):
    """Custom exception for dependency graph loading failures."""
    pass

class MLIRSplittingError(Exception):
    pass

class DependencyGraphValidationError(Exception):
    pass

# Map function names to actual functions
dispatch_split_fn = {}

def register_split_fn(func):
    dispatch_split_fn[func.__name__] = func
    return func

def split_by_opregion(module, operation_regions):
    # list of tuples to slice the operations
    # print warning if communication is enounterd.
    operation_blocks = []
    func_body = module.body.operations[0]
    all_ops = func_body.regions[0].blocks[0].operations
    for operation_region in operation_regions:
        start,end = operation_region[0], operation_region[1]
        curr_block = []
        for i,op in enumerate(all_ops):
            if i >= start and i < end:
                curr_block.append(op)
            if i >= end:
                break
        # this is more for sanity
        for op in curr_block:
            if is_communication_op(op):
                logging.warning(f'operation_region {operation_region} has Communication ops {op.name}')
        operation_blocks.append((curr_block, NodeType.COMP_NODE))
    return operation_blocks

class SeparatorPolicy(Enum):
    ISOLATED = auto()
    END_BLOCK = auto()
    START_BLOCK = auto()

    @classmethod
    def from_string(cls, src):
        src = src.strip().upper()
        if src == "ISOLATED":
            return cls.ISOLATED
        elif src == "END_BLOCK":
            return cls.END_BLOCK
        elif src == "START_BLOCK":
            return cls.START_BLOCK
        else:
            raise ValueError(f"Unknown SeparatorPolicy: {src}")

@register_split_fn
def linear_split(operations, block_lim=1024, separator=is_communication_op, separator_policy=SeparatorPolicy.ISOLATED, separator_node_type=NodeType.COMM_COLL_NODE):
    """
    Linear sequential split algorithm for MLIR operations.

    The algorithm scans operations in order, grouping computational ones into blocks of up to block_lim ops.
    Every communication op is isolated into its own singleton block.
    At the end, you get a sequence of COMP and COMM blocks in execution order,
    and a simple chain dependency graph linking them linearly.

    Args:
        operations: Sequence of MLIR operations to split
        block_lim: Maximum number of operations per COMP block (default: 1024)

    Returns:
        Tuple[List[(ops, block_type)], nx.DiGraph]: Sequential operation blocks and linear dependency graph
    """

    if isinstance(separator_node_type, str):
        separator_node_type = NodeType[separator_node_type.strip().upper()]

    if isinstance(separator, str):
        separator = op_classifiers[separator.strip()]

    if isinstance(separator_policy, str):
        separator_policy = SeparatorPolicy.from_string(separator_policy)

    operation_blocks = []
    curr_ops = []
    curr_ops_count = 0

    def flush_comp():
        nonlocal curr_ops, curr_ops_count
        if curr_ops:
            operation_blocks.append((curr_ops, NodeType.COMP_NODE))
            curr_ops, curr_ops_count = [], 0

    for operation in operations:
        if separator(operation):
            if separator_policy == SeparatorPolicy.ISOLATED:
                flush_comp()
                operation_blocks.append(([operation], separator_node_type))
            elif separator_policy == SeparatorPolicy.END_BLOCK:
                curr_ops += [operation]
                curr_ops_count += 1
                flush_comp()
            elif separator_policy == SeparatorPolicy.START_BLOCK:
                flush_comp()
                curr_ops += [operation]
                curr_ops_count += 1
            continue

        curr_ops += [operation]
        curr_ops_count += 1

        if curr_ops_count >= block_lim:
            flush_comp()

    flush_comp()
    dependency_graph = create_digraph(operation_blocks)
    return operation_blocks, dependency_graph

def merge_split(elements, module_dep_graph):
    def merge_blocks(a, b, hashes_seen=None):
        if hashes_seen is None:
            hashes_seen = {hash(op) for op in a}
        for op in b:
            hash_op = hash(op)
            if hash_op not in hashes_seen:
                a += [op]
                hashes_seen.add(hash_op)
        return a, hashes_seen

    merge_occured = True
    hashes_seen = {}
    next_to_check = set(range(len(elements)))
    while merge_occured:
        merge_occured = False
        for i in list(next_to_check):
            if elements[i] is None:
                continue

            block = None
            op_type = None

            pred_iter = module_dep_graph.predecessors(i)
            merge_parent = next(pred_iter, None)
            valid_single_parent_merge = False
            if merge_parent is not None:
                parent_block, parent_op_type = elements[merge_parent]
                if block is None:
                    block, op_type = elements[i]
                valid_single_parent_merge = next(pred_iter, None) is None and parent_op_type == op_type
                if valid_single_parent_merge:
                    merge_occured = True
                    new_block, hash_seen = merge_blocks(parent_block, block, hashes_seen.get(merge_parent))
                    hashes_seen[merge_parent] = hash_seen
                    elements[i] = None
                    next_to_check.discard(i)
                    elements[merge_parent] = (new_block, op_type)
                    module_dep_graph.add_edges_from([(merge_parent, child) for child in module_dep_graph.successors(i)])
                    module_dep_graph.remove_node(i)
                    continue

            if not valid_single_parent_merge:
                succ_iter = module_dep_graph.successors(i)
                merge_child = next(succ_iter, None)
                if merge_child is not None:
                    if block is None:
                        block, op_type = elements[i]
                    child_block, child_op_type = elements[merge_child]
                    if next(succ_iter, None) is None and child_op_type == op_type:
                        merge_occured = True
                        new_block, hash_seen = merge_blocks(block, child_block, hashes_seen.get(i))
                        hashes_seen[merge_child] = hash_seen
                        elements[i] = None
                        next_to_check.discard(i)
                        elements[merge_child] = (new_block, op_type)
                        module_dep_graph.add_edges_from([(parent, merge_child) for parent in module_dep_graph.predecessors(i)])
                        module_dep_graph.remove_node(i)

        childless_modules = []
        parentless_modules = []
        unconnected_modules = []
        for i in module_dep_graph:
            childless = next(module_dep_graph.successors(i), None) is None
            parentless = next(module_dep_graph.predecessors(i), None) is None
            unconnected = childless and parentless
            if unconnected:
                unconnected_modules.append(i)
            else:
                if childless:
                    childless_modules.append(i)
                if parentless:
                    parentless_modules.append(i)

        for i in unconnected_modules:
            block, op_type = elements[i]
            for candidate in childless_modules + parentless_modules:
                if candidate == i or elements[candidate] is None:
                    continue
                adopter_block, adopter_type = elements[candidate]
                if adopter_type != op_type:
                    continue
                new_block, hash_seen = merge_blocks(adopter_block, block, hashes_seen.get(candidate))
                hashes_seen[candidate] = hash_seen
                elements[i] = None
                module_dep_graph.remove_node(i)
                elements[candidate] = (new_block, op_type)
                break

    def move_node(module_dep_graph, new_element_map, old_i, new_i):
        if old_i == new_i:
            return module_dep_graph
        if old_i not in module_dep_graph:
            if new_i not in module_dep_graph:
                raise MLIRSplittingError("Node {} went missing".format(new_i))
            return module_dep_graph
        if new_i in module_dep_graph:
            module_dep_graph = move_node(module_dep_graph, new_element_map, new_i, new_element_map.index(new_i))
        module_dep_graph.add_node(new_i)
        module_dep_graph.add_edges_from([(parent, new_i) for parent in module_dep_graph.predecessors(old_i)] + [(new_i, child) for child in module_dep_graph.successors(old_i)])
        module_dep_graph.remove_node(old_i)
        return module_dep_graph

    elements, new_element_map = zip(*list([(elements[x], x) for x in range(len(elements)) if elements[x] is not None]))
    for old_i, new_i in [(new_element_map[x], x) for x in range(len(new_element_map))]:
        module_dep_graph = move_node(module_dep_graph, new_element_map, old_i, new_i)

    return elements, module_dep_graph

@register_split_fn
def individual_split(operations, merge=False):
    operation_elements = []
    operation_deps = []
    name_to_element_map = {}
    constants = {}
    unused_constants = {}

    for operation in operations:
        if is_return_op(operation):
            continue

        op_args = [hash(x) for x in operation.operands]
        op_names = [hash(x) for x in operation.results]

        op_type = NodeType.COMM_COLL_NODE if is_communication_op(operation) else NodeType.COMP_NODE
        is_constant = (len(op_args) == 0) or all([x in constants for x in op_args])
        for op_name in op_names:
            if not is_constant:
                if op_name in name_to_element_map:
                    raise MLIRSplittingError("Duplicate mapping of result {}".format(op_name))
                name_to_element_map[op_name] = len(operation_elements)
            else:
                constant_ops = [operation]
                for op_arg in op_args:
                    for const_op in constants[op_arg]:
                        constant_ops.insert(len(constant_ops)-1, const_op)
                if op_name in constants:
                    raise MLIRSplittingError("Duplicate mapping of constant result {}".format(op_name))
                constants[op_name] = constant_ops
        if is_constant:
            unused_constants[hash(operation)] = operation
            continue
        operation_block = [operation]
        block_deps = list()
        added_constants = list()
        for op_arg in op_args:
            if op_arg in constants:
                for constant_op in constants[op_arg]:
                    constant_op_hash = hash(constant_op)
                    if constant_op_hash not in added_constants:
                        if constant_op_hash in unused_constants:
                            del unused_constants[constant_op_hash]
                        operation_block.insert(len(operation_block)-1, constant_op)
                        added_constants.append(constant_op_hash)
            elif op_arg in name_to_element_map:
                block_deps.append(name_to_element_map[op_arg])

        operation_elements.append((operation_block, op_type))
        operation_deps.append(block_deps)

    for operation in unused_constants.values():
        added = False
        op_type = NodeType.COMM_COLL_NODE if is_communication_op(operation) else NodeType.COMP_NODE
        for block, block_op_type in reversed(operation_elements):
            if block_op_type == op_type:
                block.append(operation)
                added = True
                break
        if not added:
            operation_elements.append(([operation], op_type))

    module_dep_graph = create_digraph_with_deps(operation_elements, operation_deps)
    if merge:
        operation_elements, module_dep_graph = merge_split(operation_elements, module_dep_graph)
    return operation_elements, module_dep_graph

def create_digraph_with_deps(operation_blocks, operation_dependencies):
    dependency_graph = nx.DiGraph()
    for i in range(len(operation_blocks)):
        block_type = operation_blocks[i][1]
        if block_type == NodeType.COMM_COLL_NODE:
            dependency_graph.add_node(i, color="red", style="filled", fillcolor="lightcoral")
        else:
            dependency_graph.add_node(i, color="blue", style="filled", fillcolor="lightblue")
        for dependency in operation_dependencies[i]:
            dependency_graph.add_edge(dependency, i)

    if not nx.is_directed_acyclic_graph(dependency_graph):
        raise DependencyGraphValidationError("Dependency graph contains cycles")

    return dependency_graph

def create_digraph(operation_blocks):
    # create a directed networkx graph with len(operation_blocks) nodes
    # assign edges node[i] -> node[i+1] for  0<= i < len(operation_blocks)-1
    dependency_graph = nx.DiGraph()
    for i in range(len(operation_blocks)):
        block_type = operation_blocks[i][1]
        if block_type == NodeType.COMM_COLL_NODE:
            dependency_graph.add_node(i, color="red", style="filled", fillcolor="lightcoral")
        else:
            dependency_graph.add_node(i, color="blue", style="filled", fillcolor="lightblue")
        if i > 0:
            dependency_graph.add_edge(i, i-1)
    return dependency_graph

def parse_and_split_mlir(file_path, output_path, split_fn=linear_split, num_threads=-1, *split_args, **split_kw_args):
    """
    Parses the StableHLO MLIR file and uses the provided `split_fn` to split the module.
    Returns a list of modules and metadata for each split.

    :param file_path: Path to the input MLIR file.
    :param output_path: Path to write the output (unused here but assumed for downstream).
    :param split_fn: A function that takes a parsed module and returns a list of (ops, block_type) tuples.
    """

    mlir_parser = MLIRAnalyzer(mlir_path=file_path)
    operations = mlir_parser.operations
    num_operations = mlir_parser.op_count_expanded
    operation_blocks = []
    process = psutil.Process()

    log.info(get_str_divider())
    log.info(f"{'Input file'.ljust(20)}| {file_path}")
    log.info(f"{'Number of ops:'.ljust(20)}| {num_operations}")
    log.info(f"{'Split strategy'.ljust(20)}| {split_fn.__qualname__}")
    log.info(get_str_divider())
    operation_blocks, dep_graph = split_fn(operations, *split_args, **split_kw_args)
    if dep_graph is None:
        raise MLIRSplittingError(f"{split_fn} did not product dependency graph")

    # Save the dependency graph as DOT file
    dep_graph_path = os.path.join(output_path, "dependency_graph.dot")
    nx.drawing.nx_pydot.write_dot(dep_graph, dep_graph_path)
    log.info(f"Dependency graph saved to {dep_graph_path}")

    log.info(f"Number of splits: {len(operation_blocks)}")
    log.info("Creating new MLIR modules for each split:")
    new_op_count = 0
    empty_node_count = 0
    #empty_nodes = set()
    ctx = MLIRAnalyzer.get_ir_context()
    if ctx is None:
        raise MLIRSplittingError("Context is empty")
    root_funcs = mlir_parser.private_functions
    store_private_functions(root_funcs, output_path, context=ctx)
    G, name_to_func = mlir_parser.get_private_functions_nx_tree(root_funcs)

    ctx.allow_unregistered_dialects = True
    # iterate over all the nodes of the dep_graph
    # for each module index in a node, extract the corresponding operations
    # from operation_blocks, create a new MLIR module
    # write this new module to output directory
    # embed the MLIR module object into the node of the dep_graph (with key: mlir_module)
    if len(operation_blocks) == 0:
        raise MLIRSplittingError(f"No operation blocks found. Check the MLIR input {file_path}")

    class SplitProgress:
        progress_name = "Generating MLIR Modules"
        progress_unit = "modules"
        progress_bar = tqdm(desc=progress_name, unit=progress_unit) if sys.stdout.isatty() else None
        modules_written = 0
        total_modules_written = 0
        peak_mem_usage = 0
        cur_mem_usage = 0
        last_update_time = 0
        next_update_time = 0
        update_rate = 1_000_000_000
        start_time = time.perf_counter_ns()
        cur_time = start_time

        @classmethod
        def close(cls):
            cls.force_update()
            if cls.progress_bar is not None:
                cls.progress_bar.close()

        @classmethod
        def force_update(cls):
            cls.update(0, force_update=True)

        @classmethod
        def update(cls, modules_written, force_update=False):
            cls.cur_time = time.perf_counter_ns()
            cls.modules_written += modules_written
            cls.total_modules_written += modules_written
            if cls.cur_time > cls.next_update_time or force_update:
                cls.next_update_time = cls.cur_time + cls.update_rate
                cls.cur_mem_usage = process.memory_info().rss // 1_048_576
                if cls.cur_mem_usage > cls.peak_mem_usage:
                    cls.peak_mem_usage = cls.cur_mem_usage
                cls.writeout()
                cls.modules_written = 0

        @classmethod
        def writeout(cls):
            mem_usage_str = f"Mem cur:{cls.cur_mem_usage}MiB peak:{cls.peak_mem_usage}MiB"
            if cls.progress_bar is not None:
                cls.progress_bar.set_postfix_str(mem_usage_str)
                cls.progress_bar.update(cls.modules_written)
            else:
                modules_done_str = f"{cls.progress_name}: {cls.total_modules_written}{cls.progress_unit}"
                time_past_seconds = (cls.cur_time - cls.start_time) / 1e9
                modules_rate = round(cls.total_modules_written / time_past_seconds, 3)
                time_str = time.strftime("%H:%M:%S", time.gmtime(time_past_seconds))
                progress_str = f"{modules_done_str} [{time_str}, {modules_rate}{cls.progress_unit}/s, {mem_usage_str}]"
                log.info(progress_str)

    threading = False
    if num_threads == 0:
        pass
    elif num_threads == -1 or num_threads > 0:
        threading = True
        if num_threads == -1:
            num_threads = None
    else:
        raise MLIRSplittingError("Invalid number of threads '{}'".format(num_threads))

    executing = []

    if threading:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)

    def output_module(output_path, idx, ops, block_type, used_private_functions):
        new_module, inputs_dim, outputs_dim = create_new_module_with_operations(
            ops, used_private_functions, context=ctx
        )

        if new_module is None:
            log.warning(f"Module could not be generated {idx} {[op for op in ops]}")
            return idx, None

        mini_module = MLIRModule(
            mlir_module=new_module,
            block_type=block_type,
            parent_module=file_path
        )

        if not mini_module.is_communication_block:
            for output_path, output_body in mini_module.get_output_files(output_path).items():
                with open(output_path, "w") as f:
                    f.write(output_body)

        return idx, mini_module

    def update_numbers(mini_module_tuple, empty_nodes, new_op_count):
        idx, mini_module = mini_module_tuple
        if mini_module is None:
            empty_nodes.append(idx)
        else:
            dep_graph.nodes[idx]['mlir_module'] = mini_module
            new_op_count += mini_module.op_count_expanded
        SplitProgress.update(1)
        return empty_nodes, new_op_count

    empty_nodes = []

    def collect_private_functions(op, used_private_functions, added_func_names):
        """Collect private functions referenced by an operation."""
        nodes_to_add = set()
        if op.name == "func.call":
            node = op.attributes["callee"].value
            nodes_to_add.add(node)
        elif op.name == "stablehlo.custom_call":
            if "called_computations" in op.attributes:
                called_comps = op.attributes["called_computations"]
                for comp in called_comps:
                    node = comp.value
                    if node in name_to_func:
                        nodes_to_add.add(node)
        for node in nodes_to_add:
            funcs = {node} | nx.descendants(G, node)
            for f in funcs:
                if f not in added_func_names:
                    added_func_names.add(f)
                    used_private_functions.append(name_to_func[f].clone())

    for idx in dep_graph.nodes:
        ops, block_type = operation_blocks[idx]
        used_private_functions = []
        added_func_names = set()
        for op in ops:
            collect_private_functions(op, used_private_functions, added_func_names)
        if threading:
            executing.append(executor.submit(output_module, output_path, idx, ops, block_type, used_private_functions))
        else:
            empty_nodes, new_op_count = update_numbers(output_module(output_path, idx, ops, block_type, used_private_functions), empty_nodes, new_op_count)

    if threading:
        for future in concurrent.futures.as_completed(executing):
            empty_nodes, new_op_count = update_numbers(future.result(), empty_nodes, new_op_count)
        executor.shutdown(wait=True)

    empty_node_count = len(empty_nodes)
    for idx in empty_nodes:
        dep_graph.remove_node(idx)

    SplitProgress.close()

    log.info(f'Removed {empty_node_count} empty mlir nodes from dependency graph')
    # original module has a single return statement
    # each new module has a return statement
    if new_op_count < num_operations:
        raise MLIRSplittingError(f"Split [{len(operation_blocks)} modules] op count {new_op_count} should be more than original op count {num_operations} | Diff: {new_op_count - num_operations}")

    if not nx.is_directed_acyclic_graph(dep_graph):
        raise MLIRSplittingError("Dependency graph is not a DAG")
    # Save the dependency graph as pickle file
    save_dependency_graph(dep_graph, output_path)

    return dep_graph


def save_dependency_graph(dep_graph: nx.DiGraph, output_dir: Union[str, Path]) -> str:
    """
    Save a dependency graph with MLIRModule objects to a pickle file.

    This function serializes a NetworkX DiGraph containing MLIRModule objects
    in the node attributes to a pickle file for later restoration.

    Args:
        dep_graph (nx.DiGraph): Dependency graph with MLIRModule objects in node['mlir_module']
        output_dir (Union[str, Path]): Directory to save the pickle file

    Returns:
        str: Path to the saved pickle file

    Example:
        >>> dep_graph = parse_and_split_mlir("input.mlir", "output")
        >>> pickle_path = save_dependency_graph(dep_graph, "output")
        >>> print(f"Graph saved to {pickle_path}")
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    pickle_path = os.path.join(output_dir, "dependency_graph.pkl")

    try:
        with open(pickle_path, 'wb') as f:
            pickle.dump(dep_graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"Dependency graph pickled and saved to {pickle_path}")
        return pickle_path
    except pickle.PicklingError as e:
        log.warning(f"Failed to save dependency graph pickle: {e}")
        return ""


def load_dependency_graph(pickle_path: Union[str, Path]) -> Optional[nx.DiGraph]:
    """
    Load a dependency graph with MLIRModule objects from a pickle file.

    This function deserializes a NetworkX DiGraph containing MLIRModule objects
    from a pickle file. It handles MLIRModule class state restoration properly.

    Args:
        pickle_path (Union[str, Path]): Path to the pickle file

    Returns:
        Optional[nx.DiGraph]: Loaded dependency graph, or None if loading failed

    Raises:
        FileNotFoundError: If the pickle file doesn't exist
        pickle.PickleError: If the pickle file is corrupted or incompatible

    Example:
        >>> dep_graph = load_dependency_graph("output/dependency_graph.pkl")
        >>> if dep_graph:
        ...     print(f"Loaded graph with {len(dep_graph.nodes)} nodes")
    """

    try:
        if not os.path.isfile(pickle_path):
            raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

        with open(pickle_path, 'rb') as f:
            dep_graph = pickle.load(f)

        # Validate that we loaded a NetworkX DiGraph
        if not isinstance(dep_graph, nx.DiGraph):
            raise pickle.PickleError(f"Expected nx.DiGraph, got {type(dep_graph)}")

        # Validate that nodes contain mlir_module attributes
        for node_id in dep_graph.nodes:
            if 'mlir_module' not in dep_graph.nodes[node_id]:
                raise pickle.PickleError(f"Node {node_id} missing 'mlir_module' attribute")

            mlir_module = dep_graph.nodes[node_id]['mlir_module']
            if not isinstance(mlir_module, MLIRModule):
                raise pickle.PickleError(f"Node {node_id} 'mlir_module' is not an MLIRModule instance")

        log.info(f"Dependency graph loaded from {pickle_path} with {len(dep_graph.nodes)} nodes")
        return dep_graph
    except FileNotFoundError as e:
        raise LoadDependencyException(f"FileNotFoundError: Failed to load dependency graph from {pickle_path}: {e}") from e
    except pickle.PickleError as e:
        raise LoadDependencyException(f"PickleError: Failed to load dependency graph from {pickle_path}: {e}") from e


def validate_dependency_graph(dep_graph: nx.DiGraph) -> bool:
    """
    Validate the integrity of a dependency graph with MLIRModule objects.

    This function checks that the dependency graph is properly formed and
    all MLIRModule objects are valid.

    Args:
        dep_graph (nx.DiGraph): Dependency graph to validate

    Returns:
        bool: True if valid, False otherwise

    Example:
        >>> dep_graph = load_dependency_graph("output/dependency_graph.pkl")
        >>> if validate_dependency_graph(dep_graph):
        ...     print("Graph is valid")
    """
    # Check it's a directed graph
    if not isinstance(dep_graph, nx.DiGraph):
        raise DependencyGraphValidationError(f"Expected nx.DiGraph, got {type(dep_graph)}")

    # Check for cycles (should be a DAG)
    if not nx.is_directed_acyclic_graph(dep_graph):
        raise DependencyGraphValidationError("Dependency graph contains cycles")

    # Validate each node
    for node_id in dep_graph.nodes:
        node_data = dep_graph.nodes[node_id]

        # Check for required attributes
        if 'mlir_module' not in node_data:
            raise DependencyGraphValidationError(f"Node {node_id} missing 'mlir_module' attribute")

        mlir_module = node_data['mlir_module']
        if not isinstance(mlir_module, MLIRModule):
            raise DependencyGraphValidationError(f"Node {node_id} 'mlir_module' is not an MLIRModule instance")

        # Validate MLIRModule properties
        if not isinstance(mlir_module.block_type, NodeType) or mlir_module.block_type == NodeType.INVALID_NODE:
            raise DependencyGraphValidationError(f"Node {node_id} has invalid block_type: {mlir_module.block_type}")

    log.info(f"Dependency graph validation passed: {len(dep_graph.nodes)} nodes, {len(dep_graph.edges)} edges")
    return True

def main():
    parser = argparse.ArgumentParser(description="MLIR Splitter CLI")
    parser.add_argument("config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--split_fn", type=str, choices=list(dispatch_split_fn.keys()), default="linear_split", help="Split function to use (default: linear_split)")
    parser.add_argument("--block_lim", type=int, default=1024, help="Block size limit for linear_split (default: 1024)")
    parser.add_argument("--print_meta", action="store_true", default=False, help="Print metadata for each split module")
    parser.add_argument("--log-path", default=None, type=str, help="Output path for logging")
    parser.add_argument("--log-level", default='info', type=str, choices=get_log_levels(), help="Set log level")
    args = parser.parse_args()

    logger_basic_config(filename=args.log_path, level=args.log_level)

    with open(args.config, 'r') as config_file:
        config = json.load(config_file)

    mlir_file = config['mlir_file']
    output_path = config['output_dir']
    os.makedirs(output_path, exist_ok=True)

    print(f"Parsing MLIR file: {mlir_file}")
    print(f"Output path: {output_path}")


    split_fn = dispatch_split_fn[args.split_fn]

    split_kwargs = {}
    if args.split_fn == "linear_split":
        split_kwargs["block_lim"] = args.block_lim

    split_metadata = parse_and_split_mlir(mlir_file, output_path, split_fn, **split_kwargs)
    if args.print_meta:
        for idx, module in enumerate(split_metadata):
            print(f"Module {idx} metadata:")
            print(module[1])
            print(get_str_divider('='))


if __name__ == "__main__":
    main()
