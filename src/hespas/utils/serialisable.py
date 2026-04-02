# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
from __future__ import annotations
import sys
import json
from typing import Union, Any, ClassVar, get_origin, get_args, get_type_hints
if sys.version_info < (3, 11):
    from typing import NoReturn as Never
else:
    from typing import Never
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Serialisable:
    """
    This is the base class for generically serialisable dataclases, and handles serialisation and deserialisation.
    Users shouldn't worry about this class or modify it - they should only use its public interfaces.
    """

    def __to_dict(self, d):
        if d is None or isinstance(d, (int, float, str, bool)):
            return d
        elif isinstance(d, dict):
            return {k: self.__to_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [self.__to_dict(v) for v in d]
        elif hasattr(d, "to_dict"):
            return d.to_dict()
        raise ValueError("No idea how to to_dict() '{}'".format(d))

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the current serialisable Counter object into a dictionary.

        :returns: The current serialisable object as a dict
        """
        return {k: self.__to_dict(self.__dict__[k]) for k in self.__dataclass_fields__.keys()}

    @classmethod
    def from_dict(cls, input: dict[str, Any]) -> Serialisable:
        """
        Instantiates an object of this class from a dictionary. This function also does some type checking,
        as it is also used for processing the dict that is generated from loading the JSON representation.
        Because of this, it is stricter than the standard __init__, requiring all fields to be present
        (even if they have defaults in the dataclass implementation).

        :param input: dict to convert of an object of this class
        :returns: An instantiated object of this class
        """
        if not isinstance(input, dict):
            raise ValueError("{}.from_dict requires a dict as input".format(cls.__name__))
        for field_name, field_type in get_type_hints(cls).items():
            if get_origin(field_type) is ClassVar:
                continue
            if field_name not in input:
                raise ValueError("Field '{}' for {} is missing from input dictionary {}".format(field_name, cls.__name__, input))
            field_type_name = field_type.__name__
            if hasattr(field_type, "__origin__"): # Only do the first level of type checking - too hard otherwise
                if get_origin(field_type) is Union:
                    field_type_name = str(field_type)
                    field_type = get_args(field_type)
                else:
                    field_type = get_origin(field_type)
            if isinstance(field_type, tuple):
                field_type = tuple([get_origin(x) if hasattr(x, "__origin__") else x for x in field_type])
            if not isinstance(input[field_name], field_type):
                if isinstance(input[field_name], dict) and hasattr(field_type, "from_dict"):
                    pass
                else:
                    raise ValueError("Field '{}' requires type '{}', not '{}'".format(field_name, field_type_name, type(input[field_name]).__name__))
        return cls(**input)

    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> Serialisable:
        """
        Instantiates an object of this class from the file path given. Loads the file as JSON,
        then calls from_dict() to actually construct the object.
        This should be the standard inferface for deserialising the objects of this class.

        :param file_path: Path to the JSON file to load
        :returns: An instantiated object of this class
        """
        json_dict = None
        with open(file_path, 'r') as f:
            json_dict = json.load(f)
        return cls.from_dict(json_dict)

    @classmethod
    def from_json_str(cls, input: str) -> Serialisable:
        """
        Instantiates an object of this class from a JSON-formatted string
        Parses the string as JSON, then calls from_dict() to actually construct the object.

        :param input: JSON formatted string to parse
        :returns: An instantiated object of this class
        """
        return cls.from_dict(json.loads(input))

    def to_json(self, file_path: Union[str, Path]) -> Never:
        """
        Converts the current object to a dict, then converts that dict to JSON and writes it to a file.
        This should be the standard way of deserialising objects of this class.

        :param file_path: Path to the JSON file to write out
        """
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f)

    def to_json_str(self) -> str:
        """
        Converts the current object to a dict, then converts that dict to a JSON string.

        :returns: This object serialised as a JSON formatted string
        """
        return json.dumps(self.to_dict())
