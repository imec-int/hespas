# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
from ..utils.num_tools import conv_bool
from collections.abc import Mapping
from inspect import isclass

class InvalidConfigOptionError(Exception):
    """
    Custom exception to throw if config validation fails
    """
    pass

class ConfigOption:
    """
    Container class for configuration options that are defined within each estimator. These should be declared as class variables,
    and at instantiation time, the Estimator base class will iterate over every config option and set every one as an instance variable.
    Using these, the Estimator base class also automatically generates the __init__ arguments, the commandline arguments, and builds
    the structure of the hardware configuration dictionary.

    :param validation_func: Method to call to convert and validate the option value given, similar to 'int()'
    :param default: Default value for this option. If None, no default.
    :param optional: Whether this value must be set. If default is not None, this becomes True
    :param name: Default name. Shouldn't be set by the user, will be changed by the EstimateMeta metaclass in Estimator class creation
    :param description: User-readable description of this configuration option. Use in the production of the command line arguments
    :param top_level: Whether this configuration option is placed in a subdictionary on inheritance, or kept at the top level.
    :param is_dict: Treat this configuration option as a dict (don't descend in configuration)
    """
    def __init__(self, validation_func, *, default=None, optional=False, name="unnamed", post_validate_func=None, description=None, top_level=False, is_dict=False):
        if not callable(validation_func):
            raise ValueError("The validation method of a ConfigOption must be a callable method that returns the validated value")
        self.validation_func = validation_func
        self.default = default
        if not isinstance(optional, bool):
            raise ValueError("'optional' must be true or false")
        self.optional = conv_bool(optional) if default is None else True
        self.name = str(name)
        self.post_validate_func = post_validate_func
        if self.post_validate_func is not None and not callable(post_validate_func):
            raise ValueError("The post-validation method of a ConfigOption must be a callable method that returns the validated value")
        if description is not None:
            description += " ({})".format("optional" if self.optional else "required")
        self.description = description
        self.action = "store_true" if self.validation_func is bool or self.validation_func == conv_bool else None
        self.top_level = conv_bool(top_level)
        self.is_dict = True if isclass(validation_func) and issubclass(validation_func, Mapping) else is_dict

    def __call__(self, value):
        """
        Called on object instantiation when configuring the new Estimator object with the set of configuration
        options given at instantiation through dict or arguments.

        :param value: Configuration option value to validate
        :returns: Returns a validated configuration option variable to be assigned to the new instance
        :raises InvalidConfigOptionError: Raises an InvalidConfigOptionError if the validation failes
        """
        if value is None:
            if self.optional:
                return self.default
            else:
                raise InvalidConfigOptionError("Config option '{}' is required".format(self.name))
        try:
            ret_val = self.validation_func(value)
        except (InvalidConfigOptionError, ValueError, TypeError, FileNotFoundError) as e:
            raise InvalidConfigOptionError("Value '{}' for config option '{}' is invalid - {}".format(value, self.name, e)) from e
        if self.post_validate_func is not None:
            ret_val = self.post_validate_func(ret_val)
        return ret_val

