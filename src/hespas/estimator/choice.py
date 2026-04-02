# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
from .check_available import EstimatorNotAvailableError

existing_estimators = [
    "roofline",
    "cocossim",
    "iree",
    "xla",
    "onnxim",
    "zigzag_onnx",
]
available = {}
unavailable = {}

"""
File for the import of the Estimators. New Estimators should be added here following this pattern.
Allows for the optional import of Estimators which have unique dependencies.
"""

def add_estimator_choice(name, cls):
    """
    Function to add the new Estimator class to either the unconditionally available
    estimator_choice (used in unit tests) or the conditionally_available set.
    """
    if name in available:
        raise ValueError("Duplicate estimator '{}'".format(name))
    available[name] = (cls, cls.conditions)

def get_estimator(name):
    if name not in existing_estimators:
        raise ValueError("Perf Estimator method '{}' does not exist - choose from {}".format(name, existing_estimators))
    elif name in available:
        return available[name][0]
    elif name not in unavailable:
        try:
            if name == "roofline":
                from .roofline_estimator import RooflineEstimator
                add_estimator_choice(name, RooflineEstimator)
            elif name == "cocossim":
                from .cocossim_estimator import CocossimEstimator
                add_estimator_choice(name, CocossimEstimator)
            elif name == "iree":
                from .iree_estimator import IREEEstimator
                add_estimator_choice(name, IREEEstimator)
            elif name == "xla":
                from .xla_estimator import XLAEstimator
                add_estimator_choice(name, XLAEstimator)
            elif name == "onnxim":
                from .onnxim_estimator import ONNXIMEstimator
                add_estimator_choice(name, ONNXIMEstimator)
            elif name == "zigzag_onnx":
                from .zigzag_onnx_estimator import ZigzagONNXEstimator
                add_estimator_choice(name, ZigzagONNXEstimator)
            return available[name][0]
        except EstimatorNotAvailableError as e:
            unavailable[name] = str(e)
    if name in unavailable:
        raise EstimatorNotAvailableError(unavailable[name], name)

def estimator_available(name):
    if name not in existing_estimators:
        raise ValueError("Perf Estimator method '{}' does not exist - choose from {}".format(name, existing_estimators))
    if name in available:
        return True
    if name in unavailable:
        return False
    try:
        get_estimator(name)
        return True
    except EstimatorNotAvailableError:
        return False

def get_conditions(name):
    if estimator_available(name):
        return available[name][1]
    else:
        return unavailable[name]
