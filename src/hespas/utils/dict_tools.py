# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
from collections import UserDict
from collections.abc import Mapping
from .num_tools import pos_float

def recurse_merge(old_dict, new_dict):
    ret_dict = old_dict
    if isinstance(old_dict, Mapping) and isinstance(new_dict, Mapping):
        for ele in set(list(old_dict.keys()) + list(new_dict.keys())):
            ret_dict[ele] = recurse_merge(old_dict.get(ele), new_dict.get(ele))
    elif new_dict is not None:
        ret_dict = new_dict
    return ret_dict

def flatten_dict(d, root=None):
    ret_dict = {}
    for key, val in d.items():
        ret_key = key
        if root is not None:
            ret_key = "{}_{}".format(root, key)
        if isinstance(val, Mapping):
            ret_dict = {**ret_dict, **flatten_dict(val, root=ret_key)}
        else:
            ret_dict[ret_key] = val
    return ret_dict

def print_subdict_path(path):
    ret_str = ""
    if len(path) == 0:
        return ret_str
    else:
        ret_str = " in subdict"
        first = True
        for subdict in reversed(path):
            if first:
                ret_str += " '{}'".format(subdict)
                first = False
            else:
                ret_str += " within '{}'".format(subdict)
    return ret_str

class FloatDict(UserDict):
    def __setitem__(self, key, value):
        super().__setitem__(key, float(value))

class PositiveFloatDict(FloatDict):
    def __setitem__(self, key, value):
        super().__setitem__(key, pos_float(value))
