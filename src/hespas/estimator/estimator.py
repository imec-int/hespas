# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import json
import hashlib
import os
import sys
import argparse
import atexit
import logging
import concurrent.futures
from pathlib import Path
from typing import Any, Optional, Union
if sys.version_info < (3, 11):
    from typing import NoReturn as Never
else:
    from typing import Never
from collections.abc import Callable, Mapping

from .result import EstimatorResult, OpResult, ModuleResult
from .run_config import EstimatorSingleRunConfig
from .config_option import ConfigOption, InvalidConfigOptionError
from .estimator_meta import EstimatorMeta
from .cache import CacheElement
from ..mlir_parser.mlir_module import MLIRModule
from ..mlir_parser.mlir_parser import MLIRParser
from ..utils.dict_tools import recurse_merge, print_subdict_path
from ..utils.op_info import OpInfo
from ..utils.logging import get_log_levels, logger_basic_config
from ..event_counters.stream import CounterEventStream
from ..statistics import ContainerStatistic, SummingStatistic, IntervalStatistic

log = logging.getLogger(__name__)

def register_op_handler(op_names: Union[str | list[str]]) -> Callable[[object, OpInfo], OpResult]:
    """
    Decorator that registers the current method as a stablehlo operator handler.
    The wrapped method should take an OpInfo as the argument, and return an OpResult.

    :param op_names: Either a single string naming a StableHLO operator or a list of strings of StableHLO operator names
    :returns: Annotated method (will be used in class construction to determine the op handlers)
    """
    if isinstance(op_names, str):
        op_names = [op_names]
    def decorator(func: Callable[[object, OpInfo], OpResult]) -> Callable[[object, OpInfo], OpResult]:
        func._handled_ops = op_names
        return func
    return decorator

def register_default_op_handler(func: Callable[[object, OpInfo], OpResult]) -> Callable[[object, OpInfo], OpResult]:
    """
    Decorator that registers the current method as a stablehlo operator handler if no op handler can be found.
    The wrapped method should take an OpInfo as the argument, and return an OpResult.

    :returns: Annotated method (will be used in class construction to determine the op handlers)
    """
    func._default_op_handler = True
    return func

def register_pre_op_hook(func: Callable[[object, OpInfo], Never]) -> Callable[[object, OpInfo], Never]:
    """
    Decorator that registers the current method as a hook to call before estimating the operator
    given as the first argument. It should not return anything, but may manipulate object fields.

    :returns: Annotated method (will be used in class construction to create the list of pre-op hooks)
    """
    func._pre_op_hook = True
    return func

def register_post_op_hook(func: Callable[[object, OpInfo, OpResult], Never]) -> Callable[[object, OpInfo, OpResult], Never]:
    """
    Decorator that registers the current method as a hook to call after estimating the operator
    given as the first argument, and the result of the estion as its second argument.
    It should not return anything, but may manipulate object fields.

    :returns: Annotated method (will be used in class construction to create the list of post-op hooks)
    """
    func._post_op_hook = True
    return func

def register_pre_run_hook(func: Callable[[object, MLIRModule], Never]) -> Callable[[object, MLIRModule], Never]:
    """
    Decorator that registers the current method as a hook to call after estimating the module
    given as the first argument. It should not return anything, but may manipulate object fields.
    This hook is only called when an estimator is actually run for this module, and not when a cached
    result is returned.

    :returns: Annotated method (will be used in class construction to create the list of pre-run hooks)
    """
    func._pre_run_hook = True
    return func

def register_post_run_hook(func: Callable[[object, MLIRModule, ModuleResult], Never]) -> Callable[[object, MLIRModule, ModuleResult], Never]:
    """
    Decorator that registers the current method as a hook to call after estimating the module
    given as the first argument, and the result of the estimation run as its second argument.
    It should not return anything, but may manipulate object fields.
    This hook is only called when an estimator is actually run for this module, and not when a cached
    result is returned.

    :returns: Annotated method (will be used in class construction to create the list of post-run hooks)
    """
    func._post_run_hook = True
    return func

def register_pre_estimate_hook(func: Callable[[object, MLIRModule], Never]) -> Callable[[object, MLIRModule], Never]:
    """
    Decorator that registers the current method as a hook to call before estimating the module
    given as the first argument. It should not return anything, but may manipulate object fields.
    This hook is called for every module that is estimated, included cached results.

    :returns: Annotated method (will be used in class construction to create the list of pre-estimate hooks)
    """

    func._pre_estimate_hook = True
    return func

def register_post_estimate_hook(func: Callable[[object, MLIRModule, EstimatorResult], Never]) -> Callable[[object, MLIRModule, EstimatorResult], Never]:
    """
    Decorator that registers the current method as a hook to call after estimating the module
    given as the first argument, and the result of estimation as its second argument.
    It should not return anything, but may manipulate object fields.
    This hook is called for every module that is estimated, included cached results.

    :returns: Annotated method (will be used in class construction to create the list of post-estimate hooks)
    """
    func._post_estimate_hook = True
    return func

def register_module_metadata_hook(func: Callable[[object, ModuleResult], dict[str, Any]]) -> Callable[[object, ModuleResult], dict[str, Any]]:
    """
    Decorator that registers the current method as a hook to call to generate metadata from the given ModuleResult to be
    used in the EstimatorResult. This should return a (possibly empty) dict of metadata and it will be merged along with other
    registered metadata hooks. The statistics metadata should be specific to the module within the ModuleResult.

    :returns: Annotated method (will be used in class construction to create the list of metadata hooks)
    """
    func._module_metadata_hook = True
    return func

def register_cache_hit_hook(func: Callable[[object, MLIRModule, str], Never]) -> Callable[[object, MLIRModule, str], Never]:
    """
    Decorator that registers the current method as a hook to call when the given module and hash hit in the cache.
    It should not return anything, but may manipulate object fields.

    :returns: Annotated method (will be used in class construction to create the list of cache hit hooks)
    """
    func._cache_hit_hook = True
    return func

def register_cache_miss_hook(func: Callable[[object, MLIRModule, str], Never]) -> Callable[[object, MLIRModule, str], Never]:
    """
    Decorator that registers the current method as a hook to call when the given module and hash mises in the cache.
    It should not return anything, but may manipulate object fields.

    :returns: Annotated method (will be used in class construction to create the list of cache miss hooks)
    """
    func._cache_miss_hook = True
    return func

def register_metadata_hook(func: Callable[[object], dict[str, Any]]) -> Callable[[object], dict[str, Any]]:
    """
    Decorator that registers the current method as a hook to call to generate metadata for all the modules that have been estimated.
    This should return a (possibly empty) dict of metadata and it will be merged along with other registered metadata hooks.
    The statistics metadata should be for all modules that have been estimated.

    :returns: Annotated method (will be used in class construction to create the list of metadata hooks)
    """
    func._metadata_hook = True
    return func

def register_init_hook(func: Callable[[object, OpInfo], OpResult]) -> Callable[[object, OpInfo], OpResult]:
    """
    Decorator that registers the current method as something to be called at object initialisation.

    :returns: Annotated method (will be used in class construction tp create the list of init hooks)
    """
    func._init_hook = True
    return func

class Estimator(metaclass=EstimatorMeta):
    """
    Base class for all Estimators. Handles configuration and running of the estimation.
    Deriving classes should decend from ComputeEstimator, which contains additional functionality
    relevant to most estimators, with this class being a direct implementation of the API itself.


    When an Estimator is called to estimate the runtime of an MLIR module, there are two classes of behaviour that the Estimator will follow depending on the
    method the new class registers.

    The **@register_op_handler decorator** registers a specific method to handle one or more StableHLO operator names. This method should take a single OpInfo
    argument, and return an OpResult representing the runtime estimate of this specific operator, in the form ::

        @register_op_handler(['stablehlo.dot_general', 'stablehlo.dot'])
        def handle_dot(self, op_info: OpInfo) -> OpResult:
            ...

    If a class registers one or more op handlers, or derives from a class that does; the Estimator will use the "per-op" behaviour. This takes a list of operators
    from an MLIR module, runs registered estimators for each operator found, and returns a result based on summing the runtime of all operators.
    Deriving classes dot need, and should not, specifically override any method to achieve this; but rather just register op handling functions using the
    decorator. Deriving classes can also register a default op handler if no handler is registered for the current op by using **@register_default_op_handler**.
    If no op handlers are registered, the Estimator will instead use the "whole module" behaviour. This calls the method **_get_whole_module_result()**,
    which the deriving class should override with its own implementation, which should take a single MLIR module as its argument, and return
    a ModuleResult representing the runtime estimate of the module, in the form ::

        def _get_whole_module_result(self, module: MLIRModule) -> ModuleResult:
            ...

    These are the only methods that are required to implement Estimator behaviour in the Compute API, as most of the behaviour is contained within the base class.
    Within API are additional methods for deriving classes to further implement behaviour and configuration

    To define configuration options for the new Estimator, instead of directly implementing configuring parsing, new Estimators should use ConfigOptions within the
    class definition, like so ::

        class NewEstimator(Estimator):
            new_config_option = ConfigOption(str, description="A new configuration option with no default")
            new_default_config_option = ConfigOption(int, description="A new configuration option with a default", default=2)

    The first argument should be a "validation" function which returns the validated/converted value (like int()), then all other arguments are optional.
    If no 'default' is specified, the Estimator will throw an error if this configuration option is not given unless the argument 'optional' is set to True.

    The arguments for a ConfigOption are:
        * validation_func: Function to validate and return a validated value
        * default: Default value (Optional)
        * optional: Whether this value is optional. If a default is specified, this will default to True
        * post_validate_func: Function to be called on the value after validation occurs (Optional)
        * description: Human readable description of the config option. Use in the argparse argument parser generation
        * top_level: Whether descending classes have this in a subdictionary in configuration or at a top level this does not change the field name
          within the instantiated class, but the arguments to the call to instantiate the class and the configuration dict that is fed in.

    During initialisation, each of these options will be assigned to 'self.new_config_option' and 'self.new_default_config_option' in the instantiated object.
    Classes that derive from this new estimator will also have these configuration options automatically added. The Estimator class itself also generates
    arguments to the class initialisation itself from these, as well as parsing of the configuration dict that can be passed in, and argparse arguments
    if the estimator is used directly as a module.

    There are additional methods and decorators to help with implementing Estimator functionality:

    __post_init__()
        Estimators should not override __init__, but rather if needed define a __post_init__ method that takes no arguments to perform additional
        initialisation. This will be called Estimator's __init__ configuration is complete and the cache is loaded if enabled

    _add_event()
        Estimators should not override _add_event(), but derived classes should call it to add performance counter events to the current
        performance counter event stream, to be later written out with **output_events()**

    There are a set of 'hooks' that are called a specific points in the estimation process, usually prior or subsequent to an estimation. As these registered
    methods are instance methods of the instantiated Estimator, these have access to the object's fields and can be used to implement things like statistics
    or the generation of performance counter events. Multiple methods can be registered to occur when these hooks are called, and registered methods will be
    inherited from parent classes. The hooks will be called in the order they are registered, and in the order the parent classes are inheriented. As an example: ::

        class OtherEstimator(Estimator):
            ...
            @register_pre_run_hook
            def third_pre_run_hook(...):

        class BaseEstimator(Estimator):
            ...
            @register_pre_run_hook
            def new_pre_run_hook(...):
                ...

            @register_pre_run_hook
            def second_pre_run_hook(...):

        class ChildEstimator(BaseEstimator, OtherEstimator):
            ...
            @register_pre_run_hook
            def child_pre_run_hook(...):

    When the pre-run hooks are called for BaseEstimator, the order will be:
        1. new_pre_run_hook()
        2. second_pre_run_hook()

    And in OtherEstimator:
        1. third_pre_run_hook()

    And finally ChildEstimator:
        1. new_pre_run_hook()
        2. second_pre_run_hook()
        3. third_pre_run_hook()
        4. child_pre_run_hook()

    The current list of hooks are:

    @register_pre_op_hook
        This registers a hook that will be run before an operator estimation is made. This will only be called
        for estimators that use the per-op behaviour. The registered method should take an OpInfo as its only argument, and not return anything.

    @register_post_op_hook
        This registers a hook that will be run after an operator estimation is made. This will only be called for estimators that use the per-op
        behaviour. The registered method should take an OpInfo as its first argument, and an OpResult which represents the completed estimation
        of the operator given as its first argument as its second argument. It should not return.

    @register_pre_run_hook
        This registers a hook that will be run before a module estimation is made, but not if the estimation is cached.
        The registered method should take a MLIRModule as its only argument, and not return anything.

    @register_post_run_hook
        This registers a hook that will be run after a module estimation is made, but not if the estimation is cached.
        The registered method should take a MLIRModule as its first argument, and an ModuleResult as its second argument,
        which represents the completed estimation of the MLIR module given as its first argument. It should not return.

    @register_pre_estimate_hook
        This registers a hook that will be run before a module estimation is made, regardless of caching.
        The registered method should take a MLIRModule as its only argument, and not return anything.

    @register_post_estimate_hook
        This registers a hook that will be run after module estimation is made, regardless of caching.
        The registered method should take a MLIRModule as its first argument, and an EstimatorResult as its
        second argument, which represents the completed estimation of the MLIR module given as its first argument. 
        It should not return.

    @register_module_metadata_hook
        This registers a hook that will be run when the per-module metadata are placed into the EstimatorResult.
        This only occurs when an estimation is actually run, and not if it is cached. This specific hook runs after the pre-run
        hooks, but before the post-run hooks. It should take a ModuleResult as its only argument, and return a dict
        containing a set of metadata to place with the EstimatorResult that is generated for this run. This dict will be merged
        with all other registered hooks, in the order the hooks are registered.

    @register_cache_hit_hook
        This registers a hook that will be run on cache hits. The registered method should take a MLIRModule as its first argument,
        and the cache hash as a string for the module given as its first argument. It should not return.

    @register_cache_miss_hook
        This registers a hook that will be run on cache misses. The registered method should take a MLIRModule as its first argument,
        and the cache hash as a string for the module given as its first argument. It should not return.

    @register_metadata_hook
        This registers a hook that will be run when the public interface 'get_metadata()' is called.
        It should not take any arguments, and return a dict of metadata for all the estimations that have occured.
        This dict will be merged with all other registered hooks, in the order the hooks are registered.

    There are a set of public interface functions implemented as the common interface to interact with Estimators. Derived classes should
    avoid overriding these and keep them as the public interface, and prefer registering handlers and hooks to modify behaviour instead.
    The current set of public interfaces are:

    get_estimate()
        This is the main public interface to run an estimation. This takes an MLIRModule as its only argument to run, and triggers the entire
        estimation process. It results an EstimatorResult representing the estimation of its argument.

    write_cache()
        This is the public interface for writing the current cache state to the configured file. Does nothing if the cache is configured to be
        disabled. It should not be overriden by decending classes, but should be called if the estimator wants to specifically ensure the current
        cache state is saved.

    get_metadata()
        This is the public interface for accessing the metadata dict for the set of estimations this instantiated Estimator has performed.
        The specific content is not defined, but acts as a generic container. All registered metadata_hook functions are call when this interface
        is used, and their returned dicts are merged in order to produce the dict this method returns.

    get_statistics()
        This is the public interface for accessing the statistics tree for the set of estimations this instantiated Estimator has performed.

    output_events()
        This is the public interface to write the performance event counters that have been collected during the set of estimations that
        this instantiated Estimator has performed to a file. The specification of this file can be found in the CounterEventStream definition file.

    """
    compute_estimator = False
    conditions = None
    allow_multiprocess = False

    cache_dir = ConfigOption(str, description="Cache dir path", default="hespas_cache", top_level=True)
    disable_cache = ConfigOption(bool, description="Disable the estimator cache", default=False, top_level=True)
    in_memory_only_cache = ConfigOption(bool, description="Only use an in-memory cache, do not read or write cache files", default=False, top_level=True)
    num_npus = ConfigOption(bool, description="Number of NPUs to simulate", default=1, top_level=True)

    def __init__(self, hw_config=None, **kwargs):
        """
        Common initialisation for all Estimators. Should not be overriden, handles common configuration.

        :param hw_config: Dictionary containing the configuration for this estimator. Optional, and will be overriden by anything in kwargs
        :param kwargs: Dictionary of arguments to apply to this estimator's config. The legal arguments are defined in the class's config_arguments attribute
        """
        self.hw_config = self.__setup_config_dict(hw_config, **kwargs)
        self.metadata = {}
        self.stats_tree = ContainerStatistic("Statistics for {}".format(self.__class__.__name__),
            members = {
                "estimation_time": IntervalStatistic("Total estimation time", unit="s"),
                "uncached_time": SummingStatistic("Uncached Estimation Time", unit="s"),
                "runtime": SummingStatistic("Compute Runtime Estimation", unit="s", prefix="u"),
                "modules_processed": SummingStatistic("Total Modules Processed", value_type=int),
                "operators_processed": SummingStatistic("Operators Processed", value_type=int),
                "per_module": ContainerStatistic("Per-module statistics")
            }
        )
        self.fds = {}
        self.cache = {}
        self.event_counts = CounterEventStream
        self.__configure()
        self.cache_writer = None
        self.write_futures = {}
        self.hashes = {}
        atexit.register(self.__close_executor)
        if self.disable_cache is not True and self.in_memory_only_cache is not True:
            self.__load_cache()
        self.default_stats_filter = ["runtime"]
        self.__init_hooks()
        self.__post_init__()

    def __post_init__(self):
        """
        Overridable initialisation method for Estimators. Called after configuration occurs, but before
        loading the cache (if enabled). Use this instead of __init__().
        For per-op estimators, this can be used to setup global metadata or statistics
        """
        pass

    def __setup_config_dict(self, config_dict: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Method that combines both the arguments given to the Estimator as well as its
        pass hw_config into a final dictionary that used for configuration. Checks
        for config argument existance, but not validity. Should not be overriden nor called manually.

        :param config_dict: hw_config dictionary passed to __init__()
        :param kwargs: Override arguments passed to __init__()
        :returns: Elaborated configuration dictionary with overrides applied for use in _configure()
        """
        if config_dict is None:
            config_dict = {}
        for config_arg, config_val in kwargs.items():
            if config_arg not in self.config_arguments:
                raise InvalidConfigOptionError("{} has no config argument '{}'".format(self.__class__.__name__, config_arg))
            config_argument = self.config_arguments[config_arg]
            config_key = config_arg
            config_obj = config_argument
            merge_dict = {}
            cur_dict = merge_dict
            while isinstance(config_obj, tuple):
                new_key = config_key.replace("_{}".format(config_obj[0]), "")
                cur_dict[new_key] = {}
                cur_dict = cur_dict[new_key]
                config_key = config_obj[0]
                config_obj = config_obj[1]
            cur_dict[config_key] = config_val
            config_dict = recurse_merge(config_dict, merge_dict)
        return config_dict

    def __configure(self, config_dict=None, config_options=None, subdict_path=None):
        """
        Method that configures the current estimator. Each configuration option is checked for validity, and existence; and ensures that
        all required options are set. Since the configuration options are inherited, this method recursively evaluates and validates the sub-dictionaries
        within the current configuration dictionary. Each configuration option is set as an attribute. This method should not be overriden or
        manually called.

        :param config_dict: The current input dictionary to explore. Any values that are themselves dictionaries will be explored recursively
        :param config_options: The current set of valid options for this level.
        :param subdict_path: The current path of subdictionaries that have been traversed. Used for error printing.
        """
        if config_dict is None:
            config_dict = self.hw_config
        if config_options is None:
            config_options = self.config_options
        if subdict_path is None:
            subdict_path = []

        for subdict_name, subdict in [(k, v) for k, v in config_options.items() if isinstance(v, Mapping) and len(v) > 0]:
            self.__configure(config_dict.get(subdict_name, {}), config_options=subdict, subdict_path=subdict_path + [subdict_name])

        defaults_to_do = [x.name for x in config_options.values() if not isinstance(x, Mapping) and x.optional is True]
        required_configs = [x.name for x in config_options.values() if not isinstance(x, Mapping) and x.optional is False]
        key_order = sorted(list(config_dict.keys()), key=lambda x: self.bases_order[x] if x in self.bases_order else -1, reverse=True)
        for key in key_order:
            if key not in config_options:
                raise InvalidConfigOptionError("Unknown config option '{}'{} for estimator '{}'".format(key, print_subdict_path(subdict_path), self.display_name))
            eval_val = config_dict[key]
            if isinstance(eval_val, Mapping) and not (key in config_options and hasattr(config_options[key], "is_dict") and config_options[key].is_dict):
                self.__configure(eval_val, config_options=config_options[key], subdict_path=subdict_path + [key])
            else:
                setattr(self, key, config_options[key](eval_val))
                if key in required_configs:
                    required_configs.remove(key)
                if key in defaults_to_do:
                    defaults_to_do.remove(key)
        for default in defaults_to_do:
            setattr(self, default, config_options[default].default)
        if len(required_configs) > 0:
            raise InvalidConfigOptionError("Missing required configs {}{} for estimator '{}'".format(required_configs, print_subdict_path(subdict_path), self.display_name))

    def __get_cache_writer(self) -> Never:
        """
        Sets up the cache writer executor
        """
        if self.cache_writer is None:
            self.cache_writer = concurrent.futures.ThreadPoolExecutor()

    def __flush_outstanding_writes(self) -> Never:
        """
        Collects the outstanding cache futures and flushes them
        """
        if len(self.write_futures) < 1:
            return
        self.__get_cache_writer()
        [future.result() for future in self.write_futures.values()]
        self.write_futures = dict()

    def __close_executor(self) -> Never:
        """
        Ensure the executor for cache writes is shut down
        """
        self.__flush_outstanding_writes()
        if self.cache_writer is not None:
            self.cache_writer.shutdown(wait=True)
        self.cache_writer = None

    def __load_cache(self) -> Never:
        """
        Loads the specified cache file to populate the cache, if it exists.
        """

        self.__get_cache_writer()

        def load_cache_element(cache_path, hash):
            if os.path.isfile(cache_path):
                try:
                    with open(cache_path, "r") as f:
                        cache_dict = json.load(f)
                    cache_element = CacheElement.from_dict(cache_dict)
                    if cache_element.hw_config != self.hw_config or cache_element.estimator_name != self.display_name or cache_element.hash != hash:
                        return None
                    return cache_element
                except json.decoder.JSONDecodeError:
                    pass
            return None

        if os.path.isdir(self.cache_dir):
            log.info("Loading cached results from: '{}'".format(self.cache_dir))
            file_num = 0
            valid_files = 0
            read_futures = {self.cache_writer.submit(load_cache_element, os.path.join(self.cache_dir, cache_file), cache_file): cache_file for cache_file in os.listdir(self.cache_dir)}
            for future in concurrent.futures.as_completed(read_futures):
                file_num += 1
                cache_file = read_futures[future]
                cache_element = future.result()
                if cache_element is not None:
                    self.cache[cache_file] = cache_element
                    self.stats_tree.get_member("per_module").add_member(cache_element.result.module_file, ContainerStatistic.from_dict(cache_element.result.statistics))
                    valid_files += 1
            if valid_files > 0:
                log.info("Done ({}/{})".format(valid_files, file_num))
            else:
                log.info("No valid cache files")
        else:
            os.makedirs(self.cache_dir)
            log.info("No cached results found at: '{}'".format(self.cache_dir))

    def __write_cache(self, hash: Optional[str] = None) -> Never:
        """
        Write the current cache state to the files within the directory specified as 'cache_dir'.
        Should not be called externally or by descending classes - use the 'write_cache()' public interface instead.

        :param hash: The specific cache element to write, or the entire cache if the param is None
        """
        if len(self.cache) == 0 or self.in_memory_only_cache is True:
            return

        def write_cache_element(cache_path, element):
            with open(cache_path, "w") as f:
                f.write(json.dumps(element.to_dict(), indent=4))

        for element_hash in self.cache.keys() if hash is None else [hash]:
            if element_hash in self.write_futures:
                continue
            out_path = os.path.join(self.cache_dir, element_hash)
            if not os.path.isfile(out_path):
                self.write_futures[element_hash] = self.cache_writer.submit(write_cache_element, out_path, self.cache[element_hash])

    def __get_hash(self, module: MLIRModule) -> str:
        if module.module_file in self.hashes:
            return self.hashes[module.module_file]
        self.hashes[module.module_file] = hashlib.blake2b("{}{}".format(hash(module), self.hw_config).encode()).hexdigest()
        return self.hashes[module.module_file]

    def __create_new_cache_entry(self, module: MLIRModule, result: EstimatorResult, hash: Optional[str] = None, write: bool = True) -> EstimatorResult:
        """
        Creates a new cache entry with the current module, runtime estimate, and metadata. Write out (by default)
        Should not be overidden or called manually.

        :param module: Module that this estimate was generated for
        :param result: The estimation result generated from the Estimator
        :param hash: Hash for this combination of hardware configuration and result. Performance optimisation, will be generated if not specified.
        :returns: The estimation result that was cached
        """
        hash = self.__get_hash(module) if hash is None else hash
        self.cache[hash] = CacheElement(hw_config=self.hw_config, result=result, estimator_name=self.display_name, hash=hash)
        if write:
            self.__write_cache(hash=hash)
        return result

    def __get_op_estimate_by_name(self, op_name: str, op_info: OpInfo) -> OpResult:
        """
        This method takes a operator name and anop_info object and generates an estimation result for the Estimators that use
        per-op estimation. If there is no handler found for the specific operator, the default handler is called. The default
        handler method can be overriden by using the @register_default_op_handler decorator.
        This should not be called directly, but rather Estimators should use the @register_op_handler decorator with one or
        more operator names to register handling methods.

        :param op_name: The operator name to use to index the registered op_handlers
        :param op_info: The operator to estimate the time for.
        :returns: The estimation result for this operator
        """
        if op_name in self._op_handlers:
            return self._op_handlers[op_name](self, op_info)
        else:
            return self._default_op_handler(op_info)

    def __get_op_estimate(self, op_info: OpInfo) -> OpResult:
        """
        This method takes an op_info object and generates an estimation result for the Estimators that use per-op estimation.
        If there is no handler found for the specific operator, the default handler is called. The default handler method can
        be overriden by using the @register_default_op_handler decorator.
        This should not be called directly, but rather Estimators should use the @register_op_handler decorator with one or
        more operator names to register handling methods.

        :param op_info: The operator to estimate the time for. A specific handler for that op will be called, or the default
        :returns: The estimation result for this operator
        """
        return self.__get_op_estimate_by_name(op_info.op_name, op_info)

    def __get_op_estimates(self, op_info_list: [OpInfo]) -> [OpResult]:
        """
        This method takes a list of operators and calls _get_op_estimate() on each one.
        Users should only use the @register_op_handler decorator and not directly call this method.

        :param op_info_list: List of operator objects to produce estimations for
        :returns: A list of estimations for the input operator list
        """
        return [self.__get_op_estimate(op_info) for op_info in op_info_list]

    def __get_op_estimate_list(self, op_info_list: [OpInfo]) -> [OpResult]:
        """
        This method takes a list of operators and calls _get_op_estimate() on each one. It also calls pre-/post- hooks that
        are registered for operator estimations. It will return a list of estimation results for each operator.
        Users should only use the @register_op_handler decorator and not directly call this method.

        :param op_info_list: List of operator objects to produce estimations for
        :returns: A list of estimations for the input operator list
        """
        op_results = []
        for op_info in op_info_list:
            self.__pre_op_hooks(op_info)
            op_results.append(self.__get_op_estimate(op_info))
            self.__post_op_hooks(op_info, op_results[-1])
        return op_results

    def __get_per_op_module_result(self, module: MLIRModule) -> ModuleResult:
        """
        This method takes a module, splits it into a list of operators, and then calls _get_op_estimate_list() to generate
        a list of estimation results for each operator. It should not be called or overriden by users, rather
        Estimators should register handling methods using the @register_op_handler decorator. The list of operator
        estimations is directly use to instantiate the module result.

        :param module: Module to generate a list of operators from
        :returns: An estimation result for the whole module generated from the list of operator estimations
        """
        op_results = self.__get_op_estimate_list(module.ops_list)
        return ModuleResult.from_op_results(op_results, module)

    def __get_estimator_run_result(self, module: MLIRModule) -> EstimatorResult:
        """
        This method takes a module, and runs per-op runtime estimations if any op handlers are defined, otherwise runs the whole
        module estimate. It also runs the pre-/post- hooks that are registers for each actual (non-cached) estimation run.
        This will also call get_metadata() and get_statistics() with the resultant ModuleResult in order to populate the metadata
        and statistics fields of the EstimatorResult.
        This method should not be directly called or overriden by users, rather either _get_whole_module_result() should be overriden
        or a set of op handling methods should be registered with @register_op_handler.

        :param module: Module to generate the estimate for
        :returns: A estimation result for the input module
        """
        self.__pre_run_hooks(module)
        module_result = self.__get_per_op_module_result(module) if len(self._op_handlers) > 0 else self._get_whole_module_result(module)
        self.__post_run_hooks(module, module_result)
        return EstimatorResult.from_module_result(module_result, self.__get_module_metadata(module_result), self.__get_module_statistics(module_result))

    def __get_estimator_cached_result(self, module: MLIRModule, write: bool = True) -> EstimatorResult:
        """
        This method takes a module and checks the cache to see if a matching module estimate has already been made. If it has, return
        that, otherwise do a new estimation run for the current module and write the result to the cache. It also record both
        cache hits and misses.This method should not be called directly or overridden by users.

        :param module: Module to look for in the cache, or if not found, generate a new estimation for
        :param write: Write the cache to file each time the cache is written to
        :returns: An completed estimation result
        """
        hash = self.__get_hash(module)
        if hash in self.cache:
            result = self.cache[hash].result
            self.__cache_hit_hooks(module, hash)
        else:
            result = self.__create_new_cache_entry(module, self.__get_estimator_run_result(module), hash=hash, write=write)
            self.__cache_miss_hooks(module, hash)
        return result

    def __get_estimator_result(self, module: MLIRModule, write: bool = True) -> EstimatorResult:
        """
        This is the root method of starting an estimate. It will run the pre-/post- hooks that are run for every module, as well getting the
        estimation result for this module. If caching is enabled, it will attempt to get the result from the cache, otherwise it will
        run a new estimation every time. This method should not be overriden or directly called by users, rather _get_whole_module_estimate()
        should be overridden to specify an estimation that takes the entire module, or per-operator estimation handling method should be
        registered with @register_op_handler.

        :param module: The module to generate an estimation for
        :param write: Write the cache to file each time the cache is written to (ignored if the cache is disabled)
        """
        self.__pre_estimate_hooks(module)
        result = self.__get_estimator_run_result(module) if self.disable_cache else self.__get_estimator_cached_result(module, write=write)
        self.__post_estimate_hooks(module, result)
        return result

    def __init_hooks(self) -> Never:
        """
        Runs the init hooks that have been registered. Should not be called directly or overriden,
        rather users should use the @register_init_hook decorator

        """
        for init_hook in self._init_hooks:
            init_hook(self)

    def __pre_op_hooks(self, op_info: OpInfo) -> Never:
        """
        Runs the pre-operator estimation hooks that have been registered. Should not be called directly or overriden,
        rather users should use the @register_pre_op_hook decorator

        :param op_info: The operator that will be passed into the registered hooks
        """
        for pre_op_hook in self._pre_op_hooks:
            pre_op_hook(self, op_info)

    def __post_op_hooks(self, op_info: OpInfo, result: OpResult) -> Never:
        """
        Runs the post-operator estimation hooks that have been registered. Should not be called directly or overriden,
        rather users should use the @register_post_op_hook decorator

        :param op_info: The operator that will be passed into the registered hooks
        :param result: The result of the operator estimation
        """
        for post_op_hook in self._post_op_hooks:
            post_op_hook(self, op_info, result)

    def __pre_run_hooks(self, module: MLIRModule) -> Never:
        """
        Runs the pre-run estimation hooks that have been registered. Should not be called directly or overriden,
        rather users should use the @register_pre_run_hook decorator

        :param module: The module that will be passed into the registered hooks
        """
        for pre_run_hook in self._pre_run_hooks:
            pre_run_hook(self, module)

    def __post_run_hooks(self, module: MLIRModule, result: ModuleResult) -> Never:
        """
        Runs the post-run estimation hooks that have been registered. Should not be called directly or overriden,
        rather users should use the @register_post_run_hook decorator

        :param module: The module that will be passed into the registered hooks
        :param result: The estimation result of the run
        """
        for post_run_hook in self._post_run_hooks:
            post_run_hook(self, module, result)

    def __pre_estimate_hooks(self, module: MLIRModule) -> Never:
        """
        Runs the pre-estimate hooks that have been registered. Should not be called directly or overriden,
        rather users should use the @register_pre_estimate_hook decorator

        :param module: The module that will be passed into the registered hooks
        """
        for pre_estimate_hook in self._pre_estimate_hooks:
            pre_estimate_hook(self, module)

    def __post_estimate_hooks(self, module: MLIRModule, result: EstimatorResult) -> Never:
        """
        Runs the post-estimate hooks that have been registered. Should not be called directly or overriden,
        rather users should use the @register_post_estimate_hook decorator

        :param module: The module that will be passed into the registered hooks
        :param result: The result of the estimation
        """
        for post_estimate_hook in self._post_estimate_hooks:
            post_estimate_hook(self, module, result)

    def __cache_hit_hooks(self, module: MLIRModule, hash: str) -> Never:
        """
        Runs the cache hit hooks that have been registered. Should not be called directly or overriden,
        rather users should use the @register_cache_hit_hook decorator

        :param module: The module that will be passed into the registered hooks
        :param hash: The hash for this hardware config and module
        """
        for cache_hit_hook in self._cache_hit_hooks:
            cache_hit_hook(self, module, hash)

    def __cache_miss_hooks(self, module: MLIRModule, hash: str) -> Never:
        """
        Runs the cache miss hooks that have been registered. Should not be called directly or overriden,
        rather users should use the @register_cache_miss_hook decorator

        :param module: The module that will be passed into the registered hooks
        :param hash: The hash for this hardware config and module
        """
        for cache_miss_hook in self._cache_miss_hooks:
            cache_miss_hook(self, module, hash)

    def __get_module_metadata(self, module_result: ModuleResult) -> dict[str, Any]:
        """
        Runs the per-module metadata generatation hooks that have been registered and merges
        the results from all of the hooks. Should not be called directly or overriden,
        rather users should use the @register_module_metadata_hook decorator

        :param module_result: The module estimation result that will be passed into the registered hooks
        :returns: A dict containing the metadata for this module's estimation
        """
        ret_metadata = {}
        for metadata_hook in self._module_metadata_hooks:
            ret_metadata = recurse_merge(ret_metadata, metadata_hook(self, module_result))
        return ret_metadata

    def __get_module_statistics(self, module_result: ModuleResult) -> dict[str, Any]:
        """
        Returns the per-module stats, found in the 'per_module' subtree, for this module.

        :param module_result: The module estimation result to get the module name from
        :returns: A dict containing the statistics for this module's estimation
        """
        return self.stats_tree.get_member("per_module", module_result.module.module_file).to_dict()

    def __get_metadata(self) -> dict[str, Any]:
        """
        Runs the metadata generatation hooks that have been registered and merges
        the results from all of the hooks. Returns a dict containing the metadata for all
        modules that have been estimated. Should not be called directly or should not be overriden,
        rather users should use the @register_metadata_hook decorator

        :returns: A dict containing the metadata for all of the modules that have been estimated
        """
        ret_metadata = {}
        for metadata_hook in self._metadata_hooks:
            ret_metadata = recurse_merge(ret_metadata, metadata_hook(self))
        return ret_metadata

    def _get_whole_module_result(self, module: MLIRModule) -> ModuleResult:
        """
        This method takes a module and returns an estimation. This should be overriden by users in an Estimator that generates
        an estimation using the whole module (like the XLA estimator) to implement the estimation functionality. Besides taking
        a module as its input and returning an estimation result, the implemention is left to the user.

        :param module: Module to generate the estimate for
        :returns: An estimation result for the whole module
        """
        raise NotImplementedError

    def _add_event(self, event_name: str, timestamp: Union[int, float], timeunit: str = "", metadata: Optional[dict[str, Any]] = None) -> Never:
        """
        Interface to add a performance counter event to the current event stream. Should be called
        by descendent classes, but not overriden.

        :param event_name: String specifying the name of the event
        :param timestamp: Timestamp value of the event
        :param timeunit: Optional unit for the timestamp
        :param metadata: Optional metadata for this event
        """
        metadata = dict(metadata) if metadata is not None else {}
        self.event_counts.add_event(event_name=event_name, timestamp=timestamp, timeunit=timeunit, metadata=metadata)

    def get_metadata(self) -> dict[str, Any]:
        """
        Returns a dict containing the metadata for all modules that have been estimated.
        This is the public interface and should be called directly, but should not be overriden,
        rather users should use the @register_metadata_hook decorator

        :returns: A dict containing the metadata for all of the modules that have been estimated
        """
        return self.__get_metadata()

    def get_statistics(self) -> ContainerStatistic:
        """
        Returns the root ContainerStatistic with all statistics collected by this estimator.

        :returns: A ContainerStatistic containing the statistics for all of the modules that have been estimated
        """
        return self.stats_tree

    def write_cache(self) -> Never:
        """
        Writes the current state of the cache to the configured file if the cache has not been disabled,
        otherwise does nothing. This is the public interface for writing out the cache file and should
        be called directly, but should not be overriden.
        """
        if not self.disable_cache:
            self.__write_cache()
        self.__flush_outstanding_writes()

    def output_events(self, file_path: Union[str, Path]) -> Never:
        """
        Interface to write out the current set of performance counter events to file. Should be called by
        externally, but not overriden.

        :param file_path: Path to the file to write the event counter file to
        """
        self.event_counts.to_json(file_path)

    def get_estimate(self, module: MLIRModule) -> EstimatorResult:
        """
        Public facing interface to run an estimation on the given module. Should be called
        directly as the primary interface to run an estimation. Should not be overriden.

        :param module: The module to run an estimation for
        :returns: An EstimatorResult containing the runtime, metadata, and statistics
        """
        return self.__get_estimator_result(module, write=not self.in_memory_only_cache)

    @classmethod
    def run(cls, hw_config: dict[str, Any], mlir_file: Union[str, Path]) -> EstimatorResult:
        """
        Runs the estimator with the specified hardware config for a single MLIR file and returns the result.

        :param hw_config: Hardware configuration dict for this estimator
        :param mlir_file: MLIR module file to estimate
        :returns: An EstimatorResult containing the results of estimation
        """
        return cls(**hw_config).get_estimate(MLIRModule(mlir_path=mlir_file))

    @classmethod
    def main(cls, args: Optional[dict[str, Any]] = None) -> Never:
        """
        Universal commandline interface for running an Estimator

        :param args: An optional dict of args to use in arg parsing.
        """
        parser = argparse.ArgumentParser(description=f'{cls.display_name} Performance Estimator', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--config-file', help='Configuration file path, optional and can be overriden by arguments')
        parser.add_argument('--mlir-file', help='MLIR file path (optional, will use the field in the config_file otherwise)')
        parser.add_argument("--log-path", default=None, type=str, help="Output path for logging")
        parser.add_argument("--log-level", default='info', type=str, choices=get_log_levels(), help="Set log level")
        for config_argument_name in sorted(list(cls.config_arguments)):
            config_argument_obj = cls.config_arguments[config_argument_name]
            if isinstance(config_argument_obj, tuple):
                config_argument_obj = config_argument_obj[1]
            parser.add_argument("--{}".format(config_argument_name.replace("_", "-")), help=config_argument_obj.description, default=config_argument_obj.default, action=config_argument_obj.action)

        parsed_args = parser.parse_args(args)
        logger_basic_config(filename=parsed_args.log_path, level=parsed_args.log_level)
        config = EstimatorSingleRunConfig(**vars(parsed_args))

        result = cls.run(config.to_hw_config(), config.mlir_file)

        log.result(f'Execution of module completed: {result.runtime_estimate}s')

