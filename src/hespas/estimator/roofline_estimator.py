# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import logging
from functools import lru_cache
import math

from .compute_estimator import ComputeEstimator
from .estimator import (
    register_op_handler,
    register_post_run_hook,
    register_init_hook,
    register_pre_op_hook,
    register_post_op_hook,
    register_pre_estimate_hook,
    register_post_estimate_hook
)
from .result import OpResult
from .config_option import ConfigOption, InvalidConfigOptionError
from ..utils.num_tools import pos_int, conv_bool
from ..utils.dt_sizes import get_bit_width
from ..utils.dict_tools import PositiveFloatDict
from ..statistics import SummingStatistic, ValueStatistic, RatioStatistic, ContainerStatistic

log = logging.getLogger(__name__)

class RooflineMissingDatatypeError(Exception):
    pass

class RooflineEstimator(ComputeEstimator):

    allow_multiprocess = True

    peak_flops = ConfigOption(pos_int, description="Peak FLOPS/s of the estimator")
    memory_bandwidth = ConfigOption(pos_int, description="Peak Memory Bandwidth in Bytes/second of the estimator")
    tdp_W = ConfigOption(float, description="TDP to use for rough power estimation", default=0)
    hbm_power_ratio = ConfigOption(float, description="Memory power ratio to use for rough power estimation", default=0)
    per_datatype_flops = ConfigOption(PositiveFloatDict, description="Peak FLOPS/s dict for each datatype", optional=True)
    warn_on_unknown_type = ConfigOption(conv_bool, description="Warn if the datatype for the operation is not specified in per_datatype_flops", default=None, optional=True)
    error_on_unknown_type = ConfigOption(conv_bool, description="Error if the datatype for the operation is not specified in per_datatype_flops", default=False)

    def __post_init__(self):
        self.has_per_datatype_flops = True
        if self.per_datatype_flops is None or len(self.per_datatype_flops) == 0:
            if self.error_on_unknown_type:
                raise InvalidConfigOptionError("error_on_unknown_type specified but per_datatype_flops is empty")
            elif self.warn_on_unknown_type:
                raise log.warning("warn_on_unknown_type specified but per_datatype_flops is empty")
            self.has_per_datatype_flops = False

        if self.warn_on_unknown_type is None:
            self.warn_on_unknown_type = self.has_per_datatype_flops

    @lru_cache
    def __get_datatype_str(self, datatype):
        datatype_str = str(datatype)
        if "complex" in datatype_str:
            datatype_str = str(datatype.element_type)
        return datatype_str

    @lru_cache
    def __get_datatype_str_by_op(self, op_info):
        return self.__get_datatype_str(op_info.get_largest_type())

    @lru_cache
    def __get_flops_by_datatype_str(self, datatype_str):
        if self.has_per_datatype_flops is False or datatype_str not in self.per_datatype_flops:
            return self.peak_flops
        return self.per_datatype_flops[datatype_str]

    @lru_cache
    def __get_flops_by_datatype(self, datatype):
        if self.has_per_datatype_flops is False:
            return self.peak_flops
        return self.__get_flops_by_datatype_str(self.__get_datatype_str(datatype))

    # Ops eligible for tensor core acceleration (matmul/conv use tensor cores on GPU)
    TENSOR_CORE_OPS = frozenset({"stablehlo.dot_general", "stablehlo.convolution"})
    # Datatype promotions applied by tensor cores (e.g. f32 matmul runs at TF32 speed)
    TENSOR_CORE_PROMOTIONS = {"f32": "tf32"}

    @lru_cache
    def __get_flops_by_op(self, op_info):
        if self.has_per_datatype_flops is False:
            return self.peak_flops
        datatype_str = self.__get_datatype_str_by_op(op_info)
        # Tensor core ops may run at a promoted rate (e.g. f32 → tf32)
        if op_info.op_name in self.TENSOR_CORE_OPS and datatype_str in self.TENSOR_CORE_PROMOTIONS:
            promoted = self.TENSOR_CORE_PROMOTIONS[datatype_str]
            if promoted in self.per_datatype_flops:
                return self.__get_flops_by_datatype_str(promoted)
        if datatype_str not in self.per_datatype_flops:
            info_str = "Datatype {} not found in per_datatype_flops ({}) for op: {}".format(str(datatype_str), self.per_datatype_flops, op_info)
            if self.error_on_unknown_type:
                raise RooflineMissingDatatypeError(info_str)
            elif self.warn_on_unknown_type:
                log.warning(info_str + " - using fallback flops rate {}".format(self.peak_flops))
            return self.peak_flops
        return self.__get_flops_by_datatype_str(datatype_str)

    def __get_flops(self, op_info):
        if self.has_per_datatype_flops is False:
            return self.peak_flops
        return self.__get_flops_by_op(op_info)

    def compute_runtime(self, op_info, flops, bytes_accessed):
        flops = float(flops)
        bytes_accessed = float(bytes_accessed)
        datatype_str = self.__get_datatype_str_by_op(op_info)
        compute_time = flops / self.__get_flops(op_info) if int(flops) != 0 else 0.0
        mem_time = bytes_accessed / self.memory_bandwidth if int(bytes_accessed) != 0 else 0.0
        return OpResult(success=True, op_info=op_info, runtime_estimate=max(compute_time, mem_time), metadata={"flops": flops, "bytes_accessed": bytes_accessed, "datatype": datatype_str})

    def __add_roofline_stats(self, stats_tree):
        stats_tree.add_member("flops", SummingStatistic("Total FLOPS", value_type=int), check_exists=True)
        stats_tree.add_member("per_datatype", ContainerStatistic("Stats for each datatype"), check_exists=True)
        stats_tree.add_member("mem_bytes", SummingStatistic("Total memory traffic", value_type=int, unit="B"), check_exists=True)
        stats_tree.add_member("flops_util", RatioStatistic("FLOPS/s utilisation", show_division=False), check_exists=True)
        stats_tree.add_member("mem_bw_util", RatioStatistic("Memory bandwidth utilisation", show_division=False), check_exists=True)
        stats_tree.add_member("avg_flopss", ValueStatistic("Average FLOPS/s", unit="FLOPS/s", normalise_unit=True), check_exists=True)
        stats_tree.add_member("avg_mem_bw", ValueStatistic("Average Memory Bandwidth", unit="B/s", normalise_unit=True), check_exists=True)
        if self.tdp_W != 0 and self.hbm_power_ratio != 0:
            stats_tree.add_member("total_energy", ValueStatistic("Total energy", unit="J"), check_exists=True)

    @register_init_hook
    def __setup_roofline_stats(self):
        self.__add_roofline_stats(self.stats_tree)
        self.default_stats_filter.insert(-2, "flops_util")
        self.default_stats_filter.insert(self.default_stats_filter.index("flops_util")+1, "mem_bw_util")
        self.default_stats_filter.insert(self.default_stats_filter.index("mem_bw_util")+1, "avg_flopss")
        self.default_stats_filter.insert(self.default_stats_filter.index("avg_flopss")+1, "avg_mem_bw")
        if self.stats_tree.has_member("total_energy"):
            self.default_stats_filter.insert(-2, "total_energy")

    @register_pre_estimate_hook
    def __setup_module_roofline_stats(self, module):
        self.__add_roofline_stats(self.cur_module_stats_tree)

    @register_pre_op_hook
    def __setup_per_op_roofline_stats(self, op_info):
        op_stats_tree = self.cur_module_stats_tree.get_member("per_op", op_info.op_name)
        op_stats_tree.add_member("flops", SummingStatistic("Total FLOPs", value_type=int, value=0), check_exists=True)
        op_stats_tree.add_member("mem_bytes", SummingStatistic("Total memory traffic", value_type=int, unit="B", value=0), check_exists=True)

    @register_post_op_hook
    def __get_per_op_stats(self, op_info, result):
        op_stats_tree = self.cur_module_stats_tree.get_member("per_op", op_info.op_name)
        flops = float(result.metadata.get("flops", 0))
        mem_bytes = float(result.metadata.get("bytes_accessed", 0))
        datatype = result.metadata["datatype"]
        op_stats_tree.get_member("flops").sample(flops if flops is not None else 0)
        op_stats_tree.get_member("mem_bytes").sample(mem_bytes if mem_bytes is not None else 0)
        self.__add_per_datatype_tree(op_stats_tree, datatype)
        per_datatype_tree = op_stats_tree.get_member("per_datatype")
        datatype_tree = per_datatype_tree.get_member(datatype)
        datatype_tree.get_member("flops").sample(flops if flops is not None else 0)
        datatype_tree.get_member("runtime").sample(result.runtime_estimate if result.runtime_estimate is not None else 0)

    def __add_per_datatype_tree(self, stats_tree, datatype):
        if not stats_tree.has_member("per_datatype"):
            stats_tree.add_member("per_datatype", ContainerStatistic("Stats for each datatype"), check_exists=True)
        per_datatype_tree = stats_tree.get_member("per_datatype")
        if not per_datatype_tree.has_member(datatype):
            per_datatype_tree.add_member(datatype, ContainerStatistic("Stats for datatype {}".format(datatype), members = {
                    "flops": SummingStatistic("Total FLOPS", value_type=int),
                    "runtime": SummingStatistic("Total runtime", unit="s", prefix="u"),
                    "flops_util": RatioStatistic("FLOPS/s utilisation", show_division=False)
                }),
                check_exists=True
            )

    def __merge_lower_stats_tree(self, upper_stats_tree, lower_stats_tree):
        upper_stats_tree.sample("flops", lower_stats_tree.get_member("flops"))
        upper_stats_tree.sample("mem_bytes", lower_stats_tree.get_member("mem_bytes"))
        for datatype, datatype_stats_tree in lower_stats_tree.get_member("per_datatype").items():
            self.__add_per_datatype_tree(upper_stats_tree, datatype)
            upper_datatype_tree = upper_stats_tree.get_member("per_datatype").get_member(datatype)
            lower_datatype_tree = lower_stats_tree.get_member("per_datatype").get_member(datatype)
            for stat in ["flops", "runtime"]:
                upper_datatype_tree.get_member(stat).sample(lower_datatype_tree.get_member(stat))
            flops_rate = 0
            runtime = float(upper_datatype_tree.get_member("runtime"))
            if runtime > 0:
                flops_rate = float(upper_datatype_tree.get_member("flops")) / runtime
            upper_datatype_tree.get_member("flops_util").sample(flops_rate, self.__get_flops_by_datatype_str(datatype))

    def __get_bytes_flops(self, stats_tree):
        runtime = float(stats_tree.get_member("runtime"))
        flops_rate = 0
        flops_util_weighted_sum = 0
        mem_bw_rate = 0
        if runtime > 0:
            for datatype, datatype_stats_tree in stats_tree.get_member("per_datatype").items():
                flops_util_weighted_sum += float(datatype_stats_tree.get_member("flops_util")) * float(datatype_stats_tree.get_member("runtime"))
            flops_rate = float(stats_tree.get_member("flops")) / runtime
            mem_bw_rate = float(stats_tree.get_member("mem_bytes")) / runtime
        stats_tree.sample("flops_util", flops_util_weighted_sum, runtime)
        stats_tree.sample("mem_bw_util", mem_bw_rate, self.memory_bandwidth)
        stats_tree.sample("avg_flopss", flops_rate)
        stats_tree.sample("avg_mem_bw", mem_bw_rate)
        if stats_tree.has_member("total_energy"):
            power_W_compute = float(stats_tree.get_member("flops_util")) * self.tdp_W * (1-self.hbm_power_ratio)
            power_W_mem = float(stats_tree.get_member("mem_bw_util")) * self.tdp_W * self.hbm_power_ratio
            power_W = min(power_W_mem + power_W_compute, self.tdp_W)  # cap
            stats_tree.sample("total_energy", power_W * float(self.cur_module_stats_tree.get_member("runtime")))

    @register_post_run_hook
    def __get_module_bytes_flops(self, module, result):
        for op_stats_tree in self.cur_module_stats_tree.get_member("per_op").values():
            self.__merge_lower_stats_tree(self.cur_module_stats_tree, op_stats_tree)
        self.__get_bytes_flops(self.cur_module_stats_tree)

    @register_post_estimate_hook
    def __get_total_bytes_flops(self, module, result):
        self.__merge_lower_stats_tree(self.stats_tree, self.cur_module_stats_tree)
        self.__get_bytes_flops(self.stats_tree)

    # Operation Handlers
    @register_op_handler(['stablehlo.add', 'stablehlo.subtract', 'stablehlo.multiply', 'stablehlo.divide', \
            'stablehlo.compare', 'stablehlo.maximum', 'stablehlo.minimum', 'stablehlo.and', 'stablehlo.or', 'stablehlo.xor',
            ])
    def handle_elementwise_binary(self, op_info):
        flops = math.prod(op_info.output_types[0][0]) # shape of tensor, dtype does not matter
        total_bytes = op_info.get_input_bytes(0) + op_info.get_input_bytes(1) + op_info.get_output_bytes(0)
        return self.compute_runtime(op_info, flops, total_bytes)

    @register_op_handler(['stablehlo.clamp'])
    def handle_clamp(self, op_info):
        flops = math.prod(op_info.output_types[0][0]) # shape of tensor, dtype does not matter
        total_bytes = op_info.get_input_bytes(0) + op_info.get_input_bytes(1) + op_info.get_input_bytes(2) + op_info.get_output_bytes(0)
        return self.compute_runtime(op_info, flops, total_bytes)

    @register_op_handler(['stablehlo.constant', 'func.return', 'func.call', 'stablehlo.iota', 'stablehlo.partition_id',
            'stablehlo.replica_id', 'stablehlo.get_tuple_element', 'stablehlo.optimization_barrier', 'stablehlo.return', 'mhlo.return',
            'stablehlo.real', 'stablehlo.imag', 'stablehlo.complex'])
    def handle_free_ops(self, op_info):
        return self.compute_runtime(op_info, 0, 0)

    # TODO: transpose/broadcast_in_dim also takes indices for transpose. how to make it better? technically free
    @register_op_handler(['stablehlo.transpose', 'stablehlo.reshape', 'stablehlo.broadcast_in_dim', \
                                    'stablehlo.pad', 'stablehlo.reverse', 'stablehlo.reduce_precision', 'mhlo.bitcast', 'mhlo.copy'])
    def handle_noflop_ops(self, op_info):
        total_bytes = op_info.get_input_bytes(0) + op_info.get_output_bytes(0)
        return self.compute_runtime(op_info, 0, total_bytes)

    @register_op_handler('stablehlo.concatenate')
    def handle_concatenate(self, op_info):
        # Concatenate is a zero-FLOP, memory-bound op.
        input_bytes = sum([op_info.get_input_bytes(i) for i in range(len(op_info.input_types))])
        output_bytes = op_info.get_output_bytes(0)

        total_bytes = input_bytes + output_bytes
        return self.compute_runtime(op_info, 0, total_bytes)

    @register_op_handler('stablehlo.gather')
    def handle_gather(self, op_info):
        assert len(op_info.output_types) == 1
        data_bytes = op_info.get_input_bytes(0)              # operand
        index_bytes = op_info.get_input_bytes(1) if len(op_info.input_types) > 1 else 0  # start_indices
        out_bytes = op_info.get_output_bytes(0)
        total_bytes = data_bytes + index_bytes + out_bytes
        return self.compute_runtime(op_info, 0, total_bytes)

    @register_op_handler('stablehlo.scatter')
    def handle_scatter(self, op_info):
        assert len(op_info.output_types) == 1
        data_bytes = op_info.get_input_bytes(0)              # operand
        index_bytes = op_info.get_input_bytes(1) if len(op_info.output_types) > 1 else 0  # start_indices
        out_bytes = op_info.get_output_bytes(0)
        total_bytes = data_bytes + index_bytes + out_bytes
        return self.compute_runtime(op_info, 0, total_bytes)

    @register_op_handler('stablehlo.convolution')
    def handle_convolution(self, op_info):
        kernel_shape = op_info.input_types[1][0]  # e.g. [KH, KW, C, F]
        output_shape = op_info.output_types[0][0] # e.g. [B, OH, OW, F]

        B, OH, OW, C_out = output_shape
        KH, KW, C, _ = kernel_shape

        # FLOPs: KH x KW x C multiplications and additions per output element
        flops_per_kernel = KH * KW * C * 2  # mul + add
        total_flops = B * OH * OW * C_out * flops_per_kernel

        input_bytes = op_info.get_input_bytes(0)  # input activation
        kernel_bytes = op_info.get_input_bytes(1) # weights
        output_bytes = op_info.get_output_bytes(0)

        total_bytes = input_bytes + kernel_bytes + output_bytes
        return self.compute_runtime(op_info, total_flops, total_bytes)

    @register_op_handler(['stablehlo.abs', 'stablehlo.tanh','stablehlo.is_finite','stablehlo.convert','stablehlo.exponential',\
                                    'stablehlo.sign', 'stablehlo.log', 'stablehlo.negate', 'stablehlo.power', 'stablehlo.cosine', 'stablehlo.rsqrt',
                                    'stablehlo.sqrt', 'stablehlo.logistic', 'stablehlo.sine', 'stablehlo.round_nearest_even', 'stablehlo.not'])
    def handle_unary_elemwise(self, op_info):
        assert len(op_info.output_types)==1
        flops = math.prod(op_info.output_types[0][0])
        total_bytes = op_info.get_input_bytes(0) + op_info.get_output_bytes(0)
        return self.compute_runtime(op_info, flops, total_bytes)

    @register_op_handler('stablehlo.select')
    def handle_select(self, op_info):
        flops = math.prod(op_info.output_types[0][0]) # one compare
        total_bytes = sum([op_info.get_input_bytes(i) for i in range(3)]) + op_info.get_output_bytes(0)
        return self.compute_runtime(op_info, flops, total_bytes)

    @register_op_handler('stablehlo.reduce')
    def handle_reduce(self, op_info):
        # Calculate tensor statistics
        input_shape = op_info.input_types[0][0]
        output_shape = op_info.output_types[0][0]
        dimensions = op_info.dimensions
        reduced_size = math.prod([input_shape[d] for d in dimensions])
        output_elements = math.prod(output_shape)

        # Calculate FLOPs: (n-1) reductions per output element
        flops = (reduced_size - 1) * output_elements

        # Calculate memory access (input + output)
        input_bytes = op_info.get_input_bytes(0)
        output_bytes = op_info.get_output_bytes(0)

        total_bytes = input_bytes + output_bytes
        return self.compute_runtime(op_info, flops, total_bytes)

    @register_op_handler('stablehlo.reduce_window')
    def handle_reduce_window(self, op_info):
        # FLOPs: (window_elems - 1) reductions per output element
        output_shape = op_info.output_types[0][0]
        output_elems = int(math.prod(output_shape))

        window_dims = getattr(op_info, 'window_dimensions', None)
        window_elems = int(math.prod(window_dims)) if window_dims is not None else 1
        flops = max(window_elems - 1, 0) * output_elems

        # Memory traffic (lower bound): read input once + write output (+ init)
        input_bytes = op_info.get_input_bytes(0)
        output_bytes = op_info.get_output_bytes(0)
        init_bytes = op_info.get_input_bytes(1) if len(op_info.input_types) > 1 else 0
        total_bytes = input_bytes + output_bytes + init_bytes
        return self.compute_runtime(op_info, flops, total_bytes)

    @register_op_handler('stablehlo.select_and_scatter')
    def handle_select_and_scatter(self, op_info):
        # FIXME: Still not confident this is 100% correct - revisit

        source_shape = op_info.input_types[1][0]
        source_elems = int(math.prod(source_shape))

        window_dims = getattr(op_info, 'window_dimensions', 1)
        window_elems = int(math.prod(window_dims)) if window_dims is not None else 1
        total_flops = max(window_elems - 1, 0) * source_elems
        total_bytes = op_info.get_input_bytes(0) + op_info.get_input_bytes(1) + op_info.get_output_bytes(0)

        return self.compute_runtime(op_info, total_flops, total_bytes)

    @register_op_handler(['stablehlo.sort'])
    def handle_sort(self, op_info):
        """
        Roofline model of an operator that,
        sorts 1-dimensional slices of inputs along a dimension together.
        """
        flops = 0 # assume negligible this op is memory bound
        radix_bits_per_pass = 8 # common for gpus (assumption)
        # the op support multiple inputs but they are sorted along 1 dim together
        number_of_inputs = op_info.get_number_of_inputs()
        dtype_bits = get_bit_width(op_info.input_types[0][1])
        input_bytes = number_of_inputs * op_info.get_input_bytes(0)
        output_bytes = input_bytes
        # implementing ceiling divition by radix_bits to calculate passes.
        # each pass will correspond to a memory write for bucketisation.
        passes = max(1, (dtype_bits + radix_bits_per_pass - 1) // radix_bits_per_pass)
        total_bytes = (input_bytes + output_bytes) * passes
        return self.compute_runtime(op_info, flops, total_bytes)

    @register_op_handler(['stablehlo.dot_general', 'stablehlo.dot'])
    def handle_dot_general(self, op_info):
        """
        Calculate FLOPs for StableHLO dot_general operation.

        FLOP Calculation:
        - For each output element, we perform a dot product across contracting dimensions
        - Each dot product involves: product(contracting_dims) multiply-add operations
        - Each multiply-add = 2 FLOPs (1 multiply + 1 add)
        - Total FLOPs = 2 x product(output_shape) xx product(contracting_dimension_sizes)

        Example: Matrix multiplication A[M,K] x B[K,N] = C[M,N]
        - Output elements: M x N
        - Contracting dimension size: K
        - FLOPs = 2 x M x N x K
        """
        # Compute the product of contracting dimension sizes (not sum!)
        contracting_size = 1
        for d in op_info.dims['lhs']:  # lhs contracting dimensions
            contracting_size *= op_info.lhs_dims[d]

        # Get total number of output elements
        output_shape = op_info.output_types[0][0]
        output_size = math.prod(output_shape)

        # FLOP formula: 2 FLOPs per multiply-add, for each output element and each contracting element
        flops = 2 * output_size * contracting_size

        # Compute memory traffic (bytes)
        bytes_read = op_info.get_input_bytes(0) + op_info.get_input_bytes(1)
        bytes_written = op_info.get_output_bytes(0)  # Write output tensor
        total_bytes = bytes_read + bytes_written
        return self.compute_runtime(op_info, flops, total_bytes)

    @register_op_handler(['mhlo.ragged_dot'])
    def handle_ragged_dot(self, op_info):
        """
        Ragged dot uses a new group dimension for the ragged dimension,
        for example in mode 1 the ragged dimension is M, it will be split into G groups
        each value of the group dimension tensor corresponding to how many rows of M go to each group.
        It has 3 modes each mode considers a different dimension as the ragged one (m, k, b) respectively.
        - if mode 1 apply dot_general roofline
        - if mode 2 assume average of k = K/G (this is the only case where ragged can reduce FLOPS)
        - if mode 3 assume all batches are used, so apply dot_general roofline
        Signatures for modes:
            - 1 [b,m,k], [g,b,k,n], [b,g] -> [b,m,n]
            - 2 [b,m,k], [b,k,n],   [b,g] -> [g,b,m,n]
            - 3 [b,m,k], [b,k,n],   [g]   -> [b,m,n]
        """
        input_ranks = [len(shape) for shape, _ in op_info.input_types]
        output_ranks = [len(shape) for shape, _ in op_info.output_types]
        ranks = input_ranks + output_ranks

        M = op_info.output_types[0][0][-2]
        N = op_info.output_types[0][0][-1]
        K = op_info.input_types[0][0][-1]
        bytes_written = op_info.get_output_bytes(0)
        bytes_read = op_info.get_input_bytes(0) + op_info.get_input_bytes(1) + op_info.get_input_bytes(2)
        total_bytes = bytes_read + bytes_written

        if ranks == [3,4,2,3] or ranks == [2,3,1,2]: # Mode 1
            # from https://github.com/openxla/stablehlo/blob/6b4e61fe6325a0d49617c9d6404e94d3e8e356d6/stablehlo/transforms/ChloLegalizeToStablehlo.cpp#L2106
            # // Mode 1, where the ragged dimension is an lhs non-contracting dim (m).
            # //   lhs : [b, m, k]
            # //   rhs : [g, b, k, n]
            # //   group_sizes : [g]
            # //   result : [b, m, n]
            # // This pass basically does g iterations of [b, m, k] x [b, k, n] dot_general
            # // operations, apply partial mask of size group_sizes[i] and then add them
            # // together. This is a slow implementation that's simple enough to understand
            # // with the hope that there's already an efficient hardware kernel.
            B = 1 if output_ranks[0] == 2 else op_info.output_types[0][0][0]
            G = op_info.input_types[1][0][0] # In optimised output K_opt = K * G, so G appears to be a batching dimension
            flops = 2 * B * M * N * K * G
            # all experts read mK so the sum MK is satisfied and the each individually reads KN hence *g and ofcourse the g vector

        elif ranks == [3,3,2,4] or ranks == [2,2,1,3]:
            #G = op_info.input_types[2][0][-1]
            B = 1 if output_ranks[0] == 3 else op_info.output_types[0][0][0]
            #flops = 2 * B * M * N * (K / G)
            flops = 2 * B * M * N * K

        elif ranks == [3,3,1,3] or ranks == [2,2,1,2]:
            raise NotImplementedError("ragged_dot mode 3 not implemented")
        else:
            raise ValueError("ragged_dot mode not found")

        return self.compute_runtime(op_info, flops, total_bytes)

    @register_op_handler(['stablehlo.slice', 'stablehlo.dynamic_slice'])
    def handle_slice_ops(self, op_info):
        # For slice ops, memory traffic is output_bytes * 2
        output_bytes = op_info.get_output_bytes(0)
        total_bytes = 2 * output_bytes
        return self.compute_runtime(op_info, 0, total_bytes)

    @register_op_handler(['stablehlo.dynamic_update_slice'])
    def handle_dynamic_update_slice(self, op_info):
        # dynamic_update_slice: reads original tensor + update tensor, writes full output
        input_tensor_bytes = op_info.get_input_bytes(0)
        update_tensor_bytes = op_info.get_input_bytes(1)
        output_bytes = op_info.get_output_bytes(0)
        total_bytes = input_tensor_bytes + update_tensor_bytes + output_bytes
        return self.compute_runtime(op_info, 0, total_bytes)

    @register_op_handler(["mhlo.fusion"])
    def handle_fusion(self, op_info):
        total_bytes_accessed = sum([op_info.get_input_bytes(i) for i in range(len(op_info.input_types))]) + sum([op_info.get_output_bytes(i) for i in range(len(op_info.output_types))])
        total_flops = sum(x.metadata["flops"] for x in self._Estimator__get_op_estimates(op_info.fused_ops))
        return self.compute_runtime(op_info, total_flops, total_bytes_accessed)

    @register_op_handler(["stablehlo.custom_call"])
    def handle_custom_call(self, op_info):
        if "conv" in op_info.kernel_name:
            return self._Estimator__get_op_estimate_by_name("stablehlo.convolution", op_info)
        if "TopK" in op_info.kernel_name:
            """
            https://docs.jax.dev/en/latest/_autosummary/jax.lax.top_k.html
            TopK at least reads full input, outputs top-K values and indices.
            Algorithm details are not accounted for.
            """
            input_bytes = op_info.get_input_bytes(0)
            output_bytes = sum(op_info.get_output_bytes(i) for i in range(len(op_info.output_types)))
            total_bytes = input_bytes + output_bytes
            return self.compute_runtime(op_info, 0, total_bytes)
        log.warning("Warning: Unknown custom_call kernel '{}'".format(op_info.kernel_name))
        return self.compute_runtime(op_info, 0, 0)

if __name__ == '__main__':
    RooflineEstimator.main()

