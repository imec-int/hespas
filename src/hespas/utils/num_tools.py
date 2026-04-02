# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import sys
import statistics

fmean = statistics.fmean

if sys.version_info < (3, 11):
    def _fmean(data, weights=None):
        if weights is None:
            return statistics.fmean(data)
        sum_weight = sum(weights)
        return sum([x * y for x, y in zip(data, [z / sum_weight for z in weights])])
    fmean = _fmean

def bounded_number(num_type, val, max=None, min=None, mod=None):
    val = num_type(val)
    if max is not None and val > max:
        raise ValueError("Value of type '{}' is greater than maximum '{}'".format(num_type.__name__, max))
    elif min is not None and val < min:
        raise ValueError("Value of type '{}' is less than minimum '{}'".format(num_type.__name__, min))
    elif mod is not None and val % mod != 0:
        raise ValueError("Value of type '{}' is not a multiple of '{}'".format(num_type.__name__, mod))
    return val

def conv_bool(val):
    if val is True or val is False:
        return val
    elif isinstance(val, str):
        if val.lower() == "true":
            return True
        elif val.lower() == "false":
            return False
    raise ValueError("Invalid value '{}' of type '{}' for bool conversion".format(val, type(val).__name__))

def unsigned_int(val, max=None, mod=None):
    return bounded_number(int, val, min=0, max=max, mod=mod)

def pos_int(val, max=None, mod=None):
    return bounded_number(int, val, min=1, max=max, mod=mod)

def pos_float(val, max=None):
    return bounded_number(float, val, min=0.0, max=max)

