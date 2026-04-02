# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
from __future__ import annotations
from typing import Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from ..mlir_parser.mlir_module import MLIRModule
from ..utils.num_tools import conv_bool
from ..utils.op_info import OpInfo
from ..utils.serialisable import Serialisable

@dataclass
class Result(Serialisable):
    """
    Base class for all results. Contains common fields that all Results classes share, as well
    as adding the serialisation/deserialisation functionality through Serialisable.

    Result classes define the common container classes for the Estimators to use for the results of estimation runs.
    These are all serialisable, and can be loaded from JSON. This is used in the implementation of estimator caching.
    These classes are important to the user as depending on the type of Estimator behaviour used, either
    OpResult or ModuleResult should be returned from the implemention.
    If the whole module behaviour is used, with no op handlers registered, the new _get_whole_module_result() method
    should return a ModuleResult.
    If the per-op behaviour is used, each registered op handler should return an OpResult. This is also the container
    class passed into any registered post-op hooks.
    EstimatorResult is returned from the public 'get_estimate()' Estimator interface, as well as passed into any registered
    post_run and post_estimate hooks. Users should not have to constructor EstimatorResults themselves, as
    they are generated within the base class from the returned ModuleResult or OpResult objects in the descended
    implementation.

    :param success: A bool representing if this estimation has completed successfully or not.
    :param runtime_estimate: A time representing the estimated runtime for this estimate if successful, otherwise None.
    :param metadata: A (possibly empty) dict containing metadata specific to this estimation.
    """
    success: bool
    runtime_estimate: Optional[float]
    metadata: Optional[dict[str, Any]]

    def __post_init__(self):
        self.success = conv_bool(self.success)
        self.runtime_estimate = float(self.runtime_estimate) if self.runtime_estimate is not None else None

@dataclass
class OpResult(Result):
    """
    Class representing a per-operator estimation result. Contains the op_info that represents the operator
    that this estimation was created for, as well as the basic Result fields.

    :param success: A bool representing if this estimation has completed successfully or not.
    :param runtime_estimate: A time representing the estimated runtime for this estimate if successful, otherwise None.
    :param metadata: A (possibly empty) dict containing metadata specific to this estimation.
    :param op_info: An OpInfo object representing the operator this estimation was produced for
    """
    op_info: OpInfo

@dataclass
class ModuleResult(Result):
    """
    Class representing the result of estimation for the entire MLIR module, whether per-operator or using the
    whole module behaviour. When using the whole module behaviour, an object of this class should be directly
    returned from the new _get_whole_module_result() method. If using the per-op behaviour, an object of this
    class is instantiated by the Estimator base class using the from_op_results() class method and a list of
    returned OpResults. Contains the base Result fields, as well as the module this estimation was produced for.

    :param success: A bool representing if this estimation has completed successfully or not.
    :param runtime_estimate: A time representing the estimated runtime for this estimate if successful, otherwise None.
    :param metadata: A (possibly empty) dict containing metadata specific to this estimation.
    :param module: An MLIRModule object representing the MLIR module this estimation was produced for
    """
    module: MLIRModule

    @classmethod
    def from_op_results(cls, op_results: [OpResult], module: MLIRModule, strict: bool = False) -> ModuleResult:
        """
        This class method is used by the base Estimator class to generate a ModuleResult from a list of OpResults
        when using the per-op behaviour. It should not be called by users, rather if using the per-op behaviour,
        op handler methods should be registered as detailed in the Estimator documentation.

        :param op_results: List of OpResult objects to sum together to get the result for this module
        :param module: The module the OpResult estimations were generated for
        :param strict: If true, set that the entire module estimation failed if some of the operator estimations, rather than all.
        :returns: ModuleResult representing the estimation of the whole module from the list the per-op estimations
        """
        fail = True
        runtime_estimate = None
        metadata = {}
        for op_result in op_results:
            if op_result.metadata is not None:
                if "ops" not in metadata:
                    metadata["ops"] = []
                metadata["ops"].append(op_result.metadata)
            if strict is True and (op_result.success is False or op_result.runtime is None):
                fail = True
            elif op_result.success:
                fail = False
            if op_result.runtime_estimate is not None:
                if runtime_estimate is None:
                    runtime_estimate = op_result.runtime_estimate
                else:
                    runtime_estimate += op_result.runtime_estimate
        return cls(success=not fail, module=module, runtime_estimate=runtime_estimate, metadata=metadata)

@dataclass
class EstimatorResult(Result):
    """
    Class representing the final result of estimation for the entire MLIR module, including statistics.
    This class should not be instantiated directly, but rather will be generated from the ModuleResult
    that is either returned from the _get_whole_module_result() method if the whole module behaviour is
    used, or the ModuleResult that is creating using from_op_results() and the list of OpResults returned
    from the registed op handlers.
    Objects of this class are passed into the post-run and post-estimate hooks, allowing inspection of
    the results of an estimation after it occurs. It is also the type of the object used in the cache
    to store cached estimation results, and the type of object that the public 'get_estimate()' returns
    when called on an Estimator.
    Contains the base Result fields, as well as the module_file and module_idx for the module this estimation
    was produced for. It also contains a dict for holding the statistics of this specific estimation.
    When the EstimatorResult is produced from a ModuleResult in an Estimator, the module_statistics and
    module_metadata hooks are called to directly populate the statistics and metadata fields respectively.

    :param success: A bool representing if this estimation has completed successfully or not.
    :param runtime_estimate: A time representing the estimated runtime for this estimate if successful, otherwise None.
    :param metadata: A (possibly empty) dict containing metadata specific to this estimation.
    :param module_file: The MLIR module file this estimation was produced for
    :param module_idx: The id of the MLIR module this estimation was produced for
    """
    module_file: Union[str, Path]
    module_idx: int
    statistics: Optional[dict[str, Any]]

    @classmethod
    def from_module_result(cls, module_result: ModuleResult, metadata: Optional[dict[str, Any]], statistics: Optional[dict[str, Any]]) -> EstimatorResult:
        """
        This class method is used within the Estimator class to generate an EstimatorResult from a ModuleResult
        that is returned either by the _get_whole_module_result() method when using the whole module Estimator
        behaviour, or by ModuleResult.from_op_results() when using the per-op behaviour.
        In gerneral, this should not be called by users; rather the standard Estimator behaviours should be used.

        :param module_result: A ModuleResult representing the estimation for this module
        :param metadata: A possibly empty dict containing metadata specific to this estimation
        :param statistics: A possibly empty dict containing statistics specific to this estimation
        :returns: An EstimatorResult representing the produced estimation, its metadata, and its statistics.
        """
        return cls(success=module_result.success, runtime_estimate=module_result.runtime_estimate, module_file=module_result.module.module_file,
                   module_idx=module_result.module.idx, metadata=metadata, statistics=statistics)

