"""
MLIR Module wrapper class for handling split MLIR modules with metadata.

This module provides a convenient object-oriented interface for working with
MLIR modules that have been split from a larger module, along with their
associated metadata.
"""
# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import hashlib
import json
import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from functools import lru_cache, cached_property

from .mlir_common import comm_op_names, NodeType, parse_replica_groups
from .mlir_analyzer import MLIRAnalyzer

from ..utils.tensor_size import compute_tensor_size

class MLIRModule:
    """
    A wrapper class for MLIR module strings and their associated metadata.

    This class encapsulates an MLIR module (as a string or file path) along with metadata
    about the module, such as block type, operation count, input/output dimensions,
    and other relevant information used in the HESPAS workload analysis.

    Each module is automatically assigned a unique incrementing index starting from 0.

    Attributes:
        mlir_string (str): The MLIR module as a string representation
        idx (int): Unique incrementing index for this module
        metadata (Dict[str, Any]): Dictionary containing module metadata

    Example:
        >>> # Create from MLIR string
        >>> module = MLIRModule(
        ...     mlir_string="module { ... }",
        ...     block_type=NodeType.COMM_COLL_NODE,
        ...     parent_module="input.mlir",
        ... )
        >>> print(module.op_count)
        42
        >>> print(module.idx)
        0
        >>>
        >>> # Create from file path
        >>> module = MLIRModule(
        ...     mlir_string="/path/to/module.mlir",  # File path
        ...     block_type=1
        ... )
        >>> module.save_to_files("/output/dir")
    """

    # Class variable to track the next index to assign
    _next_idx = 0

    def __init__(self, *,
                 mlir_string: Optional[str] = None,
                 mlir_path: Optional[str|Path] = None,
                 mlir_module = None,
                 block_type: Optional[NodeType|str] = None,
                 parent_module: Optional[str] = None,
                 **kwargs):
        """
        Initialize an MLIRModule instance.

        Args:
            mlir_string (str): Either the MLIR module as a string or a file path to an MLIR file
            block_type (str, optional): Type of the block ( COMM or COMP)
            parent_module (str, optional): File name of the module from which this was split
            **kwargs: Additional metadata fields
        """

        if sum([x is not None for x in [mlir_string, mlir_path, mlir_module]]) > 1:
            raise ValueError("Only one of either mlir_string or mlir_path or mlir_module must be specified")

        if isinstance(block_type, str):
            # Compat conversions
            if block_type == "COMM":
                block_type = NodeType.COMM_COLL_NODE
            elif block_type == "COMP":
                block_type = NodeType.COMP_NODE
            else:
                block_type = NodeType[block_type]
        # Assign unique incrementing index
        self.idx = MLIRModule._next_idx
        MLIRModule._next_idx += 1

        # Determine if mlir_string is a file path or actual MLIR content
        self._module_file = f"mini_module_{str(self.idx)}.mlir"
        self._mlir_string = None
        self._analyzer = None
        if mlir_path is not None:
            # It's a file path, read the content
            with open(mlir_path, 'r') as f:
                mlir_string = f.read()
            self._module_file = mlir_path
            self._module_path = MLIRModule.get_abspath(mlir_path)
            self._analyzer = MLIRAnalyzer(mlir_string=mlir_string)
        elif mlir_string is not None:
            self._analyzer = MLIRAnalyzer(mlir_string=mlir_string)
        elif mlir_module is not None:
            self._analyzer = MLIRAnalyzer(mlir_module=mlir_module)
        else:
            raise ValueError("One of mlir_string or mlir_path or mlir_module must be specified")

        self._mlir_string = self._analyzer.mlir_string
        # Store collective operation for COMM blocks

        # Store essential instance variables directly (no metadata dict)
        self._block_type = block_type
        self._parent_module = parent_module if parent_module else ""

        # Store any additional custom metadata as instance variables
        for key, value in kwargs.items():
            setattr(self, f"_{key}", value)

    @classmethod
    def reset_index_counter(cls):
        """Reset the index counter to 0. Useful for testing."""
        cls._next_idx = 0

    @cached_property
    def comm_bytes(self):
        if len(self.output_dims_mlir) > 0:
            return sum([compute_tensor_size(ret, context=MLIRAnalyzer.get_ir_context()) for ret in self.output_dims_mlir])
        return 0

    @property
    def mlir_string(self) -> str:
        """Get the MLIR module as a string."""
        if not hasattr(self, "_mlir_string") or self._mlir_string is None:
            # If the string is empty, retrieve from file
            self._mlir_string = self.retrieve_mlir_from_file()
        return self._mlir_string

    @property
    def analyzer(self):
        if not hasattr(self, '_analyzer') or self._analyzer is None:
            self._analyzer = MLIRAnalyzer(mlir_string=self.mlir_string)
        return self._analyzer

    @property
    def block_type(self) -> NodeType:
        """Get the block type (COMM or COMP)."""
        return self._block_type

    @property
    def is_communication_block(self) -> bool:
        """Check if this is a communication block."""
        return self._block_type in [NodeType.COMM_COLL_NODE] # Later expanded?

    @property
    def is_computation_block(self) -> bool:
        """Check if this is a computation block."""
        return self._block_type == NodeType.COMP_NODE

    @cached_property
    def op_count(self) -> int:
        """Get the number of operations in this module."""
        return self.analyzer.op_count

    @cached_property
    def op_count_expanded(self) -> int:
        """Get the number of operations in this module."""
        return self.analyzer.op_count_expanded

    @cached_property
    def ops_list(self):
        return self.analyzer.ops_list

    @property
    def parent_module(self) -> Optional[str]:
        """Get the parent module file name from which this was split."""
        return self._parent_module

    @property
    def module_path(self) -> Optional[Path]:
        """Get the module file path, or None if not set.
        This variable is set either if module is create from file.
        Or if module is create from string and written to file.
        """
        return self._module_path

    @property
    def module_file(self) -> str:
        """Get the module file name."""
        return self._module_file

    @cached_property
    def collective(self) -> Optional[str]:
        """Get the collective operation name (for COMM blocks)."""
        if self.is_communication_block:  # COMM block
            for op_name in comm_op_names:
                if op_name in self.mlir_string:
                    return op_name
        return None

    @cached_property
    def replica_groups(self) -> Optional[list[list[int]]]:
        """Get the parsed replica_groups for this collective (for COMM blocks).

        Returns a list of groups, where each group is a list of device ids,
        or None if not a collective or if the collective uses a different
        attribute (e.g. collective_permute with source_target_pairs).
        """
        if self.is_communication_block and "replica_groups" in self.mlir_string:  # COMM block
            return parse_replica_groups(self.mlir_string)
        return None

    @property
    def input_dims(self) -> List[str]:
        """Get the input dimensions as string representations (computed dynamically)."""
        return [str(dim) for dim in self.input_dims_mlir]

    @property
    def output_dims(self) -> List[str]:
        """Get the output dimensions as string representations (computed dynamically)."""
        return [str(dim) for dim in self.output_dims_mlir]

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Get metadata dictionary (computed dynamically for backward compatibility).

        Note: This property exists for backward compatibility. It's recommended to use
        individual properties directly instead of accessing metadata.
        """
        return self.get_metadata_copy()

    @property
    def input_dims_mlir(self) -> List:
        """
        Get the input dimensions as MLIR RankedTensorType objects.

        If the objects don't exist (e.g., after unpickling), reconstruct them
        from the MLIR string.

        Returns:
            List: List of MLIR RankedTensorType objects
        """
        if not hasattr(self, '_input_dims_mlir') or self._input_dims_mlir is None:
            self._input_dims_mlir = self.analyzer.get_mlir_function_inputs() or []
        return self._input_dims_mlir

    @property
    def output_dims_mlir(self) -> List:
        """
        Get the output dimensions as MLIR RankedTensorType objects.

        If the objects don't exist (e.g., after unpickling), reconstruct them
        from the MLIR string.

        Returns:
            List: List of MLIR RankedTensorType objects
        """
        if not hasattr(self, '_output_dims_mlir') or self._output_dims_mlir is None:
            self._output_dims_mlir = self.analyzer.get_mlir_function_outputs() or []
        return self._output_dims_mlir

    def ensure_dir(self, output_dir):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        return output_dir

    def save_mlir(self, output_path):
        # Save MLIR file
        with open(output_path, "w") as f:
            f.write(self.mlir_string)

    def get_json_metadata(self):
        # Create metadata dynamically
        return json.dumps({
            "block_type": self.block_type,
            "input_dims": self.input_dims,
            "output_dims": self.output_dims,
            "module_file": self.module_file,
            "op_count": self.op_count,
            "parent_module": self.parent_module,
            "collective": self.collective,
            "id": self.idx,
        }, default=str, indent=4)

    def save_json(self, output_path):
        # Save metadata as JSON
        with open(output_path, "w") as f:
            f.write(self.get_json_metadata())

    def get_paths(self, output_dir, base_name=None):
        if base_name is None:
            base_name = f"mini_module_{self.idx}"

        module_path = os.path.join(output_dir, f"{base_name}.mlir")
        json_path = os.path.join(output_dir, f"{base_name}.json")

        return module_path, json_path

    def save_to_files(self, output_dir: Union[str, Path], base_name: Optional[str] = None) -> tuple:
        """
        Save the MLIR module and metadata to separate files.

        Args:
            output_dir (Union[str, Path]): Directory to save files
            base_name (str, optional): Base name for the files (without extension). If not provided, uses "mini_module_{idx}"

        Returns:
            tuple: Paths to (mlir_file, json_file)

        Example:
            >>> module.save_to_files("/output")
            ... ("/output/mini_module_0.mlir", "/output/mini_module_0.json")
        """

        self._module_path, json_path = self.get_paths(output_dir, base_name=base_name)

        self.ensure_dir(output_dir)
        self.save_mlir(self._module_path)
        self.save_json(json_path)
        return str(self.module_path), str(json_path)

    def get_output_files(self, output_dir, base_name=None):
        self.ensure_dir(output_dir)
        self._module_path, json_path = self.get_paths(output_dir, base_name=base_name)
        ret_dict = {self._module_path: self.mlir_string, json_path: self.get_json_metadata()}
        return ret_dict

    def retrieve_mlir_from_file(self) -> None:
        """
        Retrieve and load the MLIR module content from the associated file.

        This method reads the MLIR string from the file specified in the `module_file`
        attribute and returns it. It is useful for reloading the MLIR content after
        it has been cleared or modified.

        Returns:
            str: The MLIR module content as a string.

        """
        if self.module_path is None:
            raise ValueError("Module path is not set.")

        if not os.path.isfile(self.module_path):
            raise FileNotFoundError(f"MLIR file not found: {self.module_path}")

        with open(self.module_path, 'r') as f:
            return f.read()

    def update_metadata(self, **kwargs) -> None:
        """
        Update metadata fields by setting instance variables.

        Args:
            **kwargs: Metadata fields to update

        Example:
            >>> module.update_metadata(op_count=50, custom_field="value")
        """
        for key, value in kwargs.items():
            # Map common metadata keys to instance variables
            if key == "block_type":
                self._block_type = value
            elif key == "op_count":
                self._op_count = value
            elif key == "parent_module":
                self._parent_module = value
            elif key == "module_file":
                self._module_file = value
            elif key == "collective":
                self._collective = value
            else:
                # Store custom fields as private instance variables
                setattr(self, f"_{key}", value)

    def get_metadata_copy(self) -> Dict[str, Any]:
        """
        Get a copy of the metadata dictionary (computed dynamically).

        Returns:
        Dict[str, Any]: A copy of the metadata
        """
        metadata = {
            "block_type": self.block_type,
            "input_dims": self.input_dims,
            "output_dims": self.output_dims,
            "module_file": self.module_file,
            "op_count": self.op_count,
            "parent_module": self.parent_module,
            "collective": self.collective,
        }

        # Add any custom metadata fields (only simple data types)
        for attr_name in dir(self):
            if attr_name.startswith('_') and not attr_name.startswith('__'):
                # Skip the main instance variables we already included
                if attr_name not in ['_block_type', '_op_count', '_parent_module',
                                   '_module_file', '_collective', '_input_dims_mlir', '_output_dims_mlir']:
                    try:
                        value = getattr(self, attr_name)
                        # Only include simple data types, not methods or complex objects
                        if not callable(value) and not hasattr(value, '__dict__'):
                            key = attr_name[1:]  # Remove leading underscore
                            metadata[key] = value
                    except (AttributeError, TypeError):
                        continue

        return metadata

    def __str__(self) -> str:
        """String representation showing basic info about the module."""
        #block_type_str = "COMM" if self.is_communication_block else "COMP"
        return f"MLIRModule(idx={self.idx}, type={self.block_type}, ops={self.op_count}, file={self.module_file})"

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return (f"MLIRModule(idx={self.idx}, block_type={self.block_type}, op_count={self.op_count}, "
                f"parent_module='{self.parent_module}', module_file='{self.module_file}')")

    def __len__(self) -> int:
        """Return the number of operations in the module."""
        return self.op_count

    @cached_property
    def hash(self):
        return int(hashlib.blake2b(self.mlir_string.encode()).hexdigest(), 16)

    def __hash__(self) -> int:
        """
        Return hash based on MLIR string, input dimensions, and output dimensions.

        This allows MLIRModule objects to be used in sets and as dictionary keys.
        The hash is computed from the same fields used in equality comparison.
        """
        # Create hash from tuple of (mlir_string, input_dims, output_dims)
        # Use SHA-256 to compute a consistent hash of the MLIR string
        return self.hash

    def __eq__(self, other) -> bool:
        """Check equality based on MLIR string, input dimensions, and output dimensions."""
        if not isinstance(other, MLIRModule):
            return False

        # Fast comparison using hash first (if both objects are hashable)
        try:
            if hash(self) != hash(other):
                return False
        except TypeError:
            # Fall back to element-wise comparison if hashing fails
            pass

        # Compare MLIR strings
        if self.mlir_string != other.mlir_string:
            return False

        # Compare input dimensions by converting to strings
        input_dims_self = [str(x) for x in self.input_dims] if self.input_dims else []
        input_dims_other = [str(x) for x in other.input_dims] if other.input_dims else []
        if input_dims_self != input_dims_other:
            return False

        # Compare output dimensions by converting to strings
        output_dims_self = [str(x) for x in self.output_dims] if self.output_dims else []
        output_dims_other = [str(x) for x in other.output_dims] if other.output_dims else []
        if output_dims_self != output_dims_other:
            return False

        return True

    def __getstate__(self) -> Dict[str, Any]:
        """
        Prepare object for pickling.

        Excludes unpickleable objects like MLIR types and stores only essential data.

        Returns:
        Dict[str, Any]: State dictionary containing all instance data and class state
        """
        state = self.__dict__.copy()
        # Store the current class-level _next_idx for restoration
        state['_class_next_idx'] = MLIRModule._next_idx

        # Remove unpickleable MLIR objects
        state.pop('_input_dims_mlir', None)
        state.pop('_output_dims_mlir', None)
        state.pop('_analyzer', None)

        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restore object from pickle.

        Args:
        state (Dict[str, Any]): State dictionary from __getstate__
        """
        # Restore the class-level _next_idx if this instance has the highest idx
        class_next_idx = state.pop('_class_next_idx', 0)
        if class_next_idx > MLIRModule._next_idx:
            MLIRModule._next_idx = class_next_idx

        # Restore instance state
        self.__dict__.update(state)
        if isinstance(self.__dict__["_block_type"], str):
            if self.__dict__["_block_type"] == "COMM":
                self.__dict__["_block_type"] = "COMM_COLL_NODE"
            elif self.__dict__["_block_type"] == "COMP":
                self.__dict__["_block_type"] = "COMP_NODE"
            self.__dict__["_block_type"] = NodeType[self.__dict__["_block_type"]]
        # Note: We don't recreate the analyzer here to avoid unnecessary MLIR parsing
        # It will be created on-demand if needed

    @classmethod
    def get_next_idx(cls) -> int:
        """
        Get the current value of the class-level index counter.

        Returns:
            int: Current value of _next_idx
        """
        return cls._next_idx

    @classmethod
    def set_next_idx(cls, value: int) -> None:
        """
        Set the class-level index counter to a specific value.

        Args:
            value (int): New value for _next_idx
        """
        cls._next_idx = value

    @staticmethod
    def get_abspath(path):
        return os.path.join(MLIRModule.get_abspath_dir(os.path.dirname(path)), os.path.basename(path))

    @staticmethod
    @lru_cache
    def get_abspath_dir(path):
        return os.path.abspath(path)
