# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
from .config_option import ConfigOption
from ..utils.dict_tools import recurse_merge, flatten_dict

class EstimatorMeta(type):
    def __new__(cls, name, bases, namespace):
        attr_hooks = [
                ("_op_handlers", "register_op_handler", "_handled_ops", dict),
                ("_pre_op_hooks", "register_pre_op_hook", "_pre_op_hook", list),
                ("_post_op_hooks", "register_post_op_hook", "_post_op_hook", list),
                ("_pre_run_hooks", "register_pre_run_hook", "_pre_run_hook", list),
                ("_post_run_hooks", "register_post_run_hook", "_post_run_hook", list),
                ("_pre_estimate_hooks", "register_pre_estimate_hook", "_pre_estimate_hook", list),
                ("_post_estimate_hooks", "register_post_estimate_hook", "_post_estimate_hook", list),
                ("_module_metadata_hooks", "register_module_metadata_hook", "_module_metadata_hook", list),
                ("_metadata_hooks", "register_metadata_hook", "_metadata_hook", list),
                ("_cache_hit_hooks", "register_cache_hit_hook", "_cache_hit_hook", list),
                ("_cache_miss_hooks", "register_cache_miss_hook", "_cache_miss_hook", list),
                ("_init_hooks", "register_init_hook", "_init_hook", list)
            ]
        if len(bases) > 0:
            if "__init__" in namespace:
                raise ValueError("{}: Estimators should not re-define __init__(). Define __post_init__() for initialisation if needed.".format(name))
            for attr_name, decorator_name, _, _ in attr_hooks:
                if attr_name in namespace:
                    raise ValueError("{}: Estimators should not re-define {}. Use the @{} decorator instead.".format(name, attr_name, decorator_name))

        if "display_name" not in namespace or namespace["display_name"] is None:
            namespace["display_name"] = name.replace("Estimator","").lower()

        namespace["config_options"] = {}
        namespace["display_name_map"] = {}
        namespace["bases_order"] = {}
        for attr_hook_name, _, _, default_constructor in attr_hooks:
            namespace[attr_hook_name] = default_constructor()
        top_level_config_options = {}
        top_level_default_op_handler = None
        default_op_handler = None

        top_level_hooks = {}
        for obj_name, obj in namespace.items():
            if isinstance(obj, ConfigOption):
                obj.name = obj_name
                namespace["config_options"][obj_name] = obj
            elif callable(obj):
                if hasattr(obj, "_default_op_handler") and obj._default_op_handler:
                    top_level_default_op_handler = obj
                else:
                    for hook_name, _, attr_name, hook_type in attr_hooks:
                        if hasattr(obj, attr_name):
                            if hook_name not in top_level_hooks:
                                top_level_hooks[hook_name] = hook_type()
                            if hook_type is dict:
                                for sub_name in getattr(obj, attr_name):
                                    top_level_hooks[hook_name][sub_name] = obj
                            elif hook_type is list:
                                top_level_hooks[hook_name].append(obj)

        for base_order, base_class in enumerate(bases):
            for attr_hook_name, _, _, default_constructor in attr_hooks:
                if hasattr(base_class, attr_hook_name):
                    if default_constructor is dict:
                        namespace[attr_hook_name] = {**namespace[attr_hook_name], **getattr(base_class, attr_hook_name)}
                    elif default_constructor is list:
                        namespace[attr_hook_name].extend(getattr(base_class, attr_hook_name))
            if hasattr(base_class, "_default_op_handler"):
                default_op_handler = base_class._default_op_handler
            if len(base_class.config_options) > 0:
                base_name = base_class.display_name
                if base_name not in namespace["display_name_map"]:
                    namespace["display_name_map"][base_name] = base_class
                    namespace["bases_order"][base_name] = base_order
                for config_option, config_object in base_class.config_options.items():
                    if config_option not in namespace["config_options"]:
                        if isinstance(config_object, ConfigOption) and config_object.top_level is True:
                            top_level_config_options[config_option] = config_object
                        else:
                            if base_name not in namespace["config_options"]:
                                namespace["config_options"][base_name] = {}
                            namespace["config_options"][base_name][config_option] = config_object

        if top_level_default_op_handler is not None:
            default_op_handler = top_level_default_op_handler

        if len(top_level_config_options) > 0:
            namespace["config_options"] = recurse_merge(namespace["config_options"], top_level_config_options)
        namespace["config_arguments"] = flatten_dict(namespace["config_options"])
        for top_level_hook_name, top_level_hooks in top_level_hooks.items():
            if isinstance(top_level_hooks, dict):
                namespace[top_level_hook_name] = recurse_merge(namespace[top_level_hook_name], top_level_hooks)
            elif isinstance(top_level_hooks, list):
                namespace[top_level_hook_name].extend(top_level_hooks)
        if default_op_handler is not None:
            namespace["_default_op_handler"] = default_op_handler

        return super().__new__(cls, name, bases, namespace)

