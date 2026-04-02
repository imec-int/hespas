# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import sys
import time
import logging
if sys.version_info < (3, 11):
    from typing import NoReturn as Never
else:
    from typing import Never
from .estimator import (
    Estimator,
    register_pre_op_hook,
    register_post_op_hook,
    register_pre_run_hook,
    register_post_run_hook,
    register_pre_estimate_hook,
    register_post_estimate_hook,
    register_cache_hit_hook,
    register_default_op_handler
)
from .result import EstimatorResult, ModuleResult, OpResult
from .config_option import ConfigOption
from ..utils.op_info import OpInfo
from ..utils.logging import get_str_divider
from ..mlir_parser.mlir_module import MLIRModule
from ..statistics.statistics import IntervalStatistic, SummingStatistic, RatioStatistic, ContainerStatistic

log = logging.getLogger(__name__)

class InvalidOpError(Exception):
    pass

class ComputeEstimator(Estimator):
    """
    This is the base class for Estimators that should be user-facing. Most Estimators should derive from this.
    For the specification of the Estimator compute api itself, refer to the documentation in the base Estimator
    class.
    """

    compute_estimator = True
    type = ConfigOption(str, description="Type of the estimator hardware", optional=True, top_level=True)

    @register_default_op_handler
    def __unknown_op_panic(self, op_info: OpInfo) -> OpResult:
        """
        This is a default for descended classes that an exception is thrown if the operator isn't known.
        This can be overriden through the @register_default_op_handler decorator, but not directly.
        This should not be directly overidden or called manually.

        :param op_info: The operator to estimate the time of. This method will not estimate, and just throw an exception
        :returns: The result of the estimator for this operator (to match the type of operator estimators - will not return)
        :raises InvalidOpError: Raises an InvalidOpError for any unknown operator
        """
        raise InvalidOpError("Unhandled operator '{}' in estimator '{}'".format(op_info.op_name, self.display_name))

    @register_pre_estimate_hook
    def __setup_per_module_stat_tree(self, module: MLIRModule) -> Never:
        """
        Setup the per-module stats subtree for this module

        :param module: The module to be estimated
        """
        self.stats_tree.get_member("per_module").add_member(module.module_file,
            ContainerStatistic("Statistics for module {}".format(module.module_file),
                members = {
                    "runtime": SummingStatistic("Compute Runtime Estimation", unit="s", prefix="u"),
                    "estimation_time": IntervalStatistic("Module estimation time", unit="s"),
                    "uncached_time": IntervalStatistic("Uncached Estimation Time", unit="s")
                }
            ),
            check_exists=True
        )
        self.cur_module_stats_tree = self.stats_tree.get_member("per_module", module.module_file)

    @register_pre_op_hook
    def __setup_per_op_stat_tree(self, op_info: OpInfo) -> Never:
        """
        Setup the per-op stats subtree for this operator

        :param op_info: The info for the operator to be estimated
        """
        self.cur_module_stats_tree.add_member("per_op", ContainerStatistic("Per-operator statistics"), check_exists=True)
        self.cur_module_stats_tree.get_member("per_op").add_member(op_info.op_name,
            ContainerStatistic("Statistics for operator {}".format(op_info.op_name),
                members = {
                    "runtime": SummingStatistic("Compute Runtime Estimation", unit="s", prefix="u"),
                    "count": SummingStatistic("Count of number of this type of operator", value_type=int)
                }
            ),
            check_exists=True
        )

    @register_pre_estimate_hook
    def __total_estimate_start_time(self, module: MLIRModule) -> Never:
        """
        A simple, universal pre-run hook that keeps track of the start wall clock time of all estimations

        :param module: The module to be estimated
        """
        if self.stats_tree.get_member("estimation_time").start is None:
            self.stats_tree.sample("estimation_time", start=time.perf_counter())
        self.cur_module_stats_tree.sample("estimation_time", start=time.perf_counter())

    @register_post_estimate_hook
    def __setup_per_op_tree(self, _: MLIRModule, _1: EstimatorResult) -> Never:
        """
        Setup the per-op stats subtree for the estimated module, included cached modules

        :param _: The module that finished estimating (ignored)
        :param _1: The result of estimating the module (ignored)
        """
        if self.cur_module_stats_tree.has_member("per_op") and not self.stats_tree.has_member("per_op"):
            self.stats_tree.add_member("per_op", ContainerStatistic("Per-operator statistics"))
            self.stats_tree.add_member("per_op_runtime_prop", ContainerStatistic("Op time proportions", sort=True))
            self.default_stats_filter.append("per_op_runtime_prop")

    @register_post_estimate_hook
    def __get_total_estimate_time(self, module: MLIRModule, result: EstimatorResult) -> Never:
        """
        A simple, universal post-estimation hook that calculates the total wall clock and uncached time

        :param module: The module that finished estimating
        :param result: The result of estimating the module
        """
        self.stats_tree.sample("estimation_time", end=time.perf_counter())
        self.cur_module_stats_tree.sample("estimation_time", end=time.perf_counter())
        self.stats_tree.sample("uncached_time", float(self.cur_module_stats_tree.get_member("uncached_time")))

    @register_post_run_hook
    def __get_module_runtime(self, _: MLIRModule, result: ModuleResult) -> Never:
        """
        A simple, universal post-run hook that keeps track of the runtime for each module estimated

        :param _: The module that finished estimating (ignored)
        :param result: The result of estimating the module
        """
        if result.runtime_estimate is not None:
            self.cur_module_stats_tree.sample("runtime", result.runtime_estimate)

    @register_post_estimate_hook
    def __get_total_runtime(self, _: MLIRModule, _1: EstimatorResult) -> Never:
        """
        A simple, universal post-estimation hook that keeps track of the runtime for each module estimated

        :param _: The module that finished estimating (ignored)
        :param _1: The result of estimating the module (ignored)
        """
        self.stats_tree.sample("runtime", self.cur_module_stats_tree.get_member("runtime"))

    @register_post_estimate_hook
    def __count_processed(self, module: MLIRModule, _: EstimatorResult) -> Never:
        """
        A simple, universal post-estimation hook that keeps track of the number of operators and modules that have been run

        :param module: The module that has been run
        :param _: The result of estimating the module (ignored)
        """
        self.stats_tree.sample("modules_processed", 1)
        self.stats_tree.sample("operators_processed", module.op_count_expanded)

    @register_post_estimate_hook
    def __get_per_op_runtime(self, _: MLIRModule, _1: EstimatorResult) -> Never:
        """
        Collects the per-op runtime stats and populates the per_op and per_op_runtime_props stat subtrees

        :param _: Module that has finished estimating (ignored)
        :param _1: Result of estimation (ignored)
        """
        if not self.stats_tree.has_member("per_op"):
            return

        for op_name, op_stats_tree in self.cur_module_stats_tree.get_member("per_op").items():
            self.stats_tree.get_member("per_op").add_member(op_name, ContainerStatistic("Statistics for operator {}".format(op_name)), check_exists=True)
            for stat_name in ["runtime", "count"]:
                stat = op_stats_tree.get_member(stat_name)
                self.stats_tree.get_member("per_op", op_name).add_member(stat_name, SummingStatistic(stat.print_name, value_type=stat.value_type, prefix=stat.prefix, unit=stat.unit), check_exists=True)
                self.stats_tree.get_member("per_op", op_name, stat_name).sample(float(stat))
            self.stats_tree.get_member("per_op_runtime_prop").add_member(op_name,
                RatioStatistic("Proportions for operator {}".format(op_name), nom=self.stats_tree.get_member("per_op", op_name, "runtime"), denom=self.stats_tree.get_member("runtime"), show_division=False),
                check_exists=True)

    @register_pre_run_hook
    def __module_run_start_time(self, module: MLIRModule) -> Never:
        """
        A simple, universal pre-run hook that keeps track of the start wall clock time of a module that will be run

        :param module: The module to be run
        """
        self.cur_module_stats_tree.sample("uncached_time", start=time.perf_counter())

    @register_post_run_hook
    def __module_run_end_time(self, module: MLIRModule, result: ModuleResult) -> Never:
        """
        A simple, universal post-run hook that keeps track of the end wall clock time and total wall clock time for
        the ran module

        :param module: The module that has been run
        :param result: The result of estimating the module
        """
        self.cur_module_stats_tree.sample("uncached_time", end=time.perf_counter())

    @register_pre_estimate_hook
    def __print_start_line(self, _: MLIRModule) -> Never:
        """
        Prints the dividing start line

        :param _: The module to be estimated (ignored)
        """
        log.progress(get_str_divider())

    @register_cache_hit_hook
    def __count_cache_hits(self, module: MLIRModule, hash: str) -> Never:
        """
        A simple, universal cache hit hook that keeps track of the number of total and per-hash cache hits

        :param module: The module that has hit in cache
        :param hash: Hash of the module that has hit in cache
        """
        if not self.stats_tree.has_member("cache_hits"):
            self.stats_tree.add_member("cache_hits", SummingStatistic("Cache hits", value_type=int))
            self.default_stats_filter.insert(0, "cache_hits")
            self.stats_tree.add_member("cache_hit_prop", RatioStatistic("Basic block reuse rate", nom=self.stats_tree.get_member("cache_hits"), denom=self.stats_tree.get_member("modules_processed")))
            self.default_stats_filter.insert(1, "cache_hit_prop")
        if not self.stats_tree.has_member("cache_op_hits"):
            self.stats_tree.add_member("cache_op_hits", SummingStatistic("Cache hits per-op", value_type=int))
            self.stats_tree.add_member("cache_op_hit_prop", RatioStatistic("Operator reuse factor", nom=self.stats_tree.get_member("cache_op_hits"), denom=self.stats_tree.get_member("operators_processed")))
            self.default_stats_filter.insert(2, "cache_op_hit_prop")
        if not self.stats_tree.has_member("total_vs_uncached"):
            self.stats_tree.add_member("total_vs_uncached", ContainerStatistic("Total estimator time taken vs cached time",
                members = {
                    "total_estimation_time": self.stats_tree.get_member("estimation_time"),
                    "total_uncached_time": self.stats_tree.get_member("uncached_time"),
                    "cached_time_reduction": RatioStatistic("Cached time proportion",
                        nom=self.stats_tree.get_member("estimation_time"),
                        denom=self.stats_tree.get_member("uncached_time"),
                        unit="%", show_division=False, value_mult=100, sub_from_one=True)
                },
                print_string="{total_estimation_time} vs total time (without caching): {total_uncached_time} ({cached_time_reduction} reduction)"
                )
            )
            self.default_stats_filter.insert(3, "total_vs_uncached")
        self.stats_tree.sample("cache_hits", 1)
        self.stats_tree.sample("cache_op_hits", module.op_count_expanded)

    @register_cache_hit_hook
    def __get_cached_module_times(self, _: MLIRModule, hash: str) -> Never:
        """
        A simple, universal cache hit hook that sets up the statistics for the module that hit in cache

        :param _: The module that hit in cache (ignored)
        :param hash: Hash of the module that has miss in cache
        """
        for cached_member_name, cached_member in self.stats_tree.get_member("per_module", self.cache[hash].result.module_file).items():
            self.cur_module_stats_tree.members[cached_member_name] = cached_member

    @register_post_op_hook
    def __get__module_op_times(self, op_info: OpInfo, result: OpResult) -> Never:
        """
        A simple, universal post-op estimation hook to collect the total op times for this module

        :param op_info: The operator estimated
        :param result: The result of the estimation of this operator
        """
        self.cur_module_stats_tree.get_member("per_op", op_info.op_name, "count").sample(1)
        if result.runtime_estimate is not None:
            self.cur_module_stats_tree.get_member("per_op", op_info.op_name, "runtime").sample(result.runtime_estimate)

    @register_cache_hit_hook
    def __print_cached_runtime(self, module: MLIRModule, hash: str) -> Never:
        """
        A simple, universal cache hit hook that prints the runtime of the cached module

        :param module: The module that hit in cache
        :param hash: Hash of the module that has miss in cache
        """
        log.progress("Cache hit ({}) for {} - runtime: {}".format(self.cache[hash].result.module_file, module.module_file, self.cache[hash].result.runtime_estimate))

    @register_post_run_hook
    def __print_run_runtime(self, module: MLIRModule, result: ModuleResult) -> Never:
        """
        A simple, universal cache hit hook that prints the runtime of the module that is actually run

        :param module: The module that hit in cache
        :param result: Result of the estimation
        """
        log.progress("Runtime estimate for {} - runtime: {}".format(module.module_file, result.runtime_estimate))

