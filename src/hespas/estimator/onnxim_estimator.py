# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import json
import os
import re
import shutil
import subprocess
import logging
from typing import Any, Dict, List
import numpy as np
from .check_available import EstimatorNotAvailableError
try:
    import onnx
    from onnx import checker
    from ..stablehlo_to_onnx.translator_api import stablehlo_ops_to_onnx_model
except ImportError as e:
    raise EstimatorNotAvailableError("Unable to import onnx: '{}'".format(e), "onnx") from e
from .estimator import register_op_handler
from .config_option import ConfigOption
from .result import OpResult
from .roofline_estimator import RooflineEstimator

log = logging.getLogger(__name__)

def _make_onnxim_initializer(value_info):
    """Build a dense placeholder initializer that ONNXim can treat as stationary data."""
    tensor_type = value_info.type.tensor_type
    elem_type = tensor_type.elem_type
    dims = [
        int(dim.dim_value) if dim.HasField("dim_value") and dim.dim_value > 0 else 1
        for dim in tensor_type.shape.dim
    ]

    if elem_type == onnx.TensorProto.BFLOAT16:
        count = int(np.prod(dims)) if dims else 1
        raw_data = np.full(count, 0x3F80, dtype=np.uint16).tobytes()
        return onnx.helper.make_tensor(
            name=value_info.name,
            data_type=onnx.TensorProto.BFLOAT16,
            dims=dims,
            vals=raw_data,
            raw=True,
        )

    np_dtype = onnx.helper.tensor_dtype_to_np_dtype(elem_type)
    values = np.ones(dims if dims else (), dtype=np_dtype)
    return onnx.numpy_helper.from_array(values, name=value_info.name)


def _make_value_info_like(value_info, *, name=None, shape=None):
    """Clone a tensor ValueInfo while allowing the caller to override its name or shape."""
    tensor_type = value_info.type.tensor_type
    dims = shape
    if dims is None:
        dims = [
            int(dim.dim_value) if dim.HasField("dim_value") and dim.dim_value > 0 else 1
            for dim in tensor_type.shape.dim
        ]
    return onnx.helper.make_tensor_value_info(
        name or value_info.name,
        tensor_type.elem_type,
        list(dims),
    )


def _value_info_dims(value_info):
    """Return a ValueInfo tensor shape as a plain list, replacing unknown dims with 1."""
    return [
        int(dim.dim_value) if dim.HasField("dim_value") and dim.dim_value > 0 else 1
        for dim in value_info.type.tensor_type.shape.dim
    ]


def _get_node_attr_ints(node, name):
    """Fetch an ONNX node attribute as a Python int list."""
    for attr in node.attribute:
        if attr.name == name:
            return list(onnx.helper.get_attribute_value(attr))
    return None


def _ceil_div(value, divisor):
    """Integer ceil division used by the mapping generator."""
    return (int(value) + int(divisor) - 1) // int(divisor)


def _get_square_conv_mapping_inputs(model):
    """Extract the Conv geometry needed for mapping generation if the model is a simple square Conv."""
    graph = model.graph
    if len(graph.node) != 1 or graph.node[0].op_type != "Conv":
        return None
    if len(graph.input) != 1 or len(graph.initializer) != 1 or len(graph.output) != 1:
        return None

    conv = graph.node[0]
    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in conv.attribute}
    strides = list(attrs.get("strides", []))
    pads = list(attrs.get("pads", []))
    kernel_shape = list(attrs.get("kernel_shape", []))
    if len(strides) != 2 or len(pads) != 4 or len(kernel_shape) != 2:
        return None
    if strides[0] != strides[1] or pads[0] != pads[1] or pads[2] != pads[3]:
        return None
    if kernel_shape[0] != kernel_shape[1]:
        return None

    input_shape = _value_info_dims(graph.input[0])
    output_shape = _value_info_dims(graph.output[0])
    weight_shape = list(graph.initializer[0].dims)
    if len(input_shape) != 4 or len(output_shape) != 4 or len(weight_shape) != 4:
        return None

    return {
        "stride": int(strides[0]),
        "padding": int(pads[0]),
        "total_n": int(input_shape[0]),
        "total_c": int(weight_shape[1]),
        "total_m": int(weight_shape[0]),
        "total_p": int(output_shape[1]),
        "total_q": int(output_shape[2]),
        "total_s": int(weight_shape[2]),
        "total_r": int(weight_shape[3]),
    }


def _get_onnxim_core_limits(device_config):
    """Derive the SRAM and accumulator limits that constrain ONNXim Conv tiling."""
    core0 = device_config["core_config"]["core_0"]
    dim = int(core0["core_height"])
    precision = int(device_config["precision"])
    return {
        "dim": dim,
        "precision": precision,
        "spad_bytes_limit": (int(core0["spad_size"]) * 1024) // 2,
        "accum_bytes_limit": (int(core0["accum_spad_size"]) * 1024) // 2,
        "max_spad_rows": (int(core0["spad_size"]) * 1024) // (dim * precision * 2),
        "max_acc_rows": (int(core0["accum_spad_size"]) * 1024) // (dim * 4 * 2),
    }


def materialize_onnxim_initializers(model):
    """
    Replace weight-side graph inputs with dense initializers so ONNXim sees stationary tensors.
    """
    graph = model.graph
    graph_inputs = {value.name: value for value in graph.input}
    initializer_names = {init.name for init in graph.initializer}
    producers = {
        output_name: node
        for node in graph.node
        for output_name in node.output
        if output_name
    }

    weight_roots = []
    for node in graph.node:
        if node.op_type in {"Conv", "ConvTranspose", "MatMul", "Gemm"}:
            weight_roots.extend(name for name in node.input[1:] if name)

    to_materialize = set()
    visited = set()

    def visit(name):
        if not name or name in visited or name in initializer_names:
            return
        visited.add(name)
        if name in graph_inputs:
            to_materialize.add(name)
            return

        producer = producers.get(name)
        if producer is None:
            return
        for input_name in producer.input:
            visit(input_name)

    for name in weight_roots:
        visit(name)

    for name in to_materialize:
        if name not in initializer_names:
            graph.initializer.append(_make_onnxim_initializer(graph_inputs[name]))

    retained_inputs = [value for value in graph.input if value.name not in to_materialize]
    del graph.input[:]
    graph.input.extend(retained_inputs)
    return model


def fold_onnxim_convolution_layout_transposes(model):
    """
    Rewrite the translator's Transpose-Conv-Transpose pattern into a single ONNXim-native Conv.
    """
    graph = model.graph
    producers = {
        output_name: node
        for node in graph.node
        for output_name in node.output
        if output_name
    }
    consumers = {}
    for node in graph.node:
        for input_name in node.input:
            if input_name:
                consumers.setdefault(input_name, []).append(node)

    initializers = {init.name: init for init in graph.initializer}
    inputs = {value.name: value for value in graph.input}

    nodes_to_remove = set()
    changed = False

    for conv in list(graph.node):
        if conv.op_type != "Conv" or len(conv.input) < 2 or len(conv.output) != 1:
            continue

        input_transpose = producers.get(conv.input[0])
        weight_transpose = producers.get(conv.input[1])
        conv_consumers = consumers.get(conv.output[0], [])

        if input_transpose is None or input_transpose.op_type != "Transpose":
            continue
        if weight_transpose is None or weight_transpose.op_type != "Transpose":
            continue
        if len(conv_consumers) != 1 or conv_consumers[0].op_type != "Transpose":
            continue

        output_transpose = conv_consumers[0]
        input_perm = _get_node_attr_ints(input_transpose, "perm")
        weight_perm = _get_node_attr_ints(weight_transpose, "perm")
        output_perm = _get_node_attr_ints(output_transpose, "perm")

        if input_perm != [0, 3, 1, 2]:
            continue
        if weight_perm != [3, 2, 0, 1]:
            continue
        if output_perm != [0, 2, 3, 1]:
            continue

        weight_source = weight_transpose.input[0]
        initializer = initializers.get(weight_source)
        if initializer is None:
            continue

        if weight_source in inputs:
            source_vi = inputs[weight_source]
            new_shape = [
                int(source_vi.type.tensor_type.shape.dim[index].dim_value)
                if source_vi.type.tensor_type.shape.dim[index].HasField("dim_value")
                and source_vi.type.tensor_type.shape.dim[index].dim_value > 0
                else 1
                for index in weight_perm
            ]
            inputs[weight_source].CopyFrom(
                _make_value_info_like(source_vi, shape=new_shape)
            )

        original_dims = list(initializer.dims)
        permuted_dims = [original_dims[index] for index in weight_perm]
        replacement = _make_onnxim_initializer(
            onnx.helper.make_tensor_value_info(
                weight_source,
                initializer.data_type,
                permuted_dims,
            )
        )
        initializer.CopyFrom(replacement)

        conv.input[0] = input_transpose.input[0]
        conv.input[1] = weight_source
        conv.output[0] = output_transpose.output[0]

        nodes_to_remove.update(
            {
                id(input_transpose),
                id(weight_transpose),
                id(output_transpose),
            }
        )
        changed = True

    if not changed:
        return model

    kept_nodes = [node for node in graph.node if id(node) not in nodes_to_remove]
    del graph.node[:]
    graph.node.extend(kept_nodes)
    return model


def prepare_onnxim_model(model):
    """Apply the ONNXim-specific graph adaptation pass sequence to a translated ONNX model."""
    model = materialize_onnxim_initializers(model)
    model = fold_onnxim_convolution_layout_transposes(model)
    return model


def create_onnxim_conv_mapping_line(model, device_config):
    """
    Recreate ONNXim's fallback Conv mapping for a single-op model while honoring the real batch size.
    """
    conv_inputs = _get_square_conv_mapping_inputs(model)
    if conv_inputs is None:
        return None
    limits = _get_onnxim_core_limits(device_config)
    stride = conv_inputs["stride"]
    padding = conv_inputs["padding"]
    dim = limits["dim"]
    precision = limits["precision"]

    def calc_rows(acc, args):
        batches, out_rows, out_cols, out_channels, kernel_rows, kernel_cols, in_channels = args
        input_rows = out_rows * stride + kernel_rows - 1
        input_cols = out_cols * stride + kernel_cols - 1
        in_channels_per_bank = _ceil_div(in_channels, dim)
        out_channels_per_bank = _ceil_div(out_channels, dim)
        a_rows = in_channels_per_bank * batches * input_rows * input_cols
        b_rows = out_channels_per_bank * kernel_cols * kernel_rows * in_channels
        c_rows = out_channels_per_bank * batches * out_rows * out_cols
        return c_rows if acc else a_rows + b_rows

    def fits(args):
        return (
            calc_rows(False, args) <= limits["max_spad_rows"]
            and calc_rows(True, args) <= limits["max_acc_rows"]
        )

    args = [
        conv_inputs["total_n"],
        conv_inputs["total_q"],
        conv_inputs["total_p"],
        conv_inputs["total_m"],
        conv_inputs["total_s"],
        conv_inputs["total_r"],
        conv_inputs["total_c"],
    ]
    max_args = list(args)
    out_rows_idx = 1
    out_cols_idx = 2
    out_channels_idx = 3
    in_channels_idx = 6

    while not fits(args):
        _, max_idx = max(
            (value, index)
            for index, value in enumerate(args)
            if not (index == out_cols_idx and args[index] <= dim and args[out_rows_idx] > 1)
        )
        if max_idx in (out_channels_idx, in_channels_idx):
            if args[max_idx] % dim != 0:
                args[max_idx] = (args[max_idx] // dim) * dim
            else:
                args[max_idx] -= dim
            args[max_idx] = max(args[max_idx], 1)
        else:
            args[max_idx] = max(args[max_idx] - 1, 1)

    while True:
        candidate = list(args)
        candidate[out_cols_idx] += 1
        if candidate[out_cols_idx] > max_args[out_cols_idx]:
            break
        if fits(candidate):
            args = candidate
        else:
            break

    while True:
        increased = False
        for index in range(len(args)):
            candidate = list(args)
            candidate[index] += 1
            if candidate[index] > max_args[index]:
                continue

            input_tile_bytes = (
                args[0]
                * (args[1] + 2 * padding)
                * (args[2] + 2 * padding)
                * args[6]
                * precision
            )
            weight_tile_bytes = args[0] * args[3] * args[4] * args[5] * args[6] * precision
            if (
                fits(candidate)
                and (((input_tile_bytes + weight_tile_bytes) * 3) >> 1) <= limits["spad_bytes_limit"]
            ):
                args = candidate
                increased = True
        if not increased:
            break

    inner_n, inner_q, inner_p, inner_m, inner_s, inner_r, inner_c = args
    if inner_n <= 0 or inner_c <= 0 or inner_m <= 0 or inner_p <= 0 or inner_q <= 0:
        return None

    output_tile_bytes = inner_n * inner_m * inner_p * inner_q * 4
    if output_tile_bytes > limits["accum_bytes_limit"]:
        return None

    outer_n = _ceil_div(conv_inputs["total_n"], inner_n)
    outer_c = _ceil_div(conv_inputs["total_c"], inner_c)
    outer_m = _ceil_div(conv_inputs["total_m"], inner_m)
    outer_s = _ceil_div(conv_inputs["total_s"], inner_s)
    outer_r = _ceil_div(conv_inputs["total_r"], inner_r)
    outer_p = _ceil_div(conv_inputs["total_p"], inner_p)
    outer_q = _ceil_div(conv_inputs["total_q"], inner_q)

    return (
        f"[T] N{conv_inputs['total_n']} C{conv_inputs['total_c']} M{conv_inputs['total_m']} "
        f"P{conv_inputs['total_p']} Q{conv_inputs['total_q']} S{conv_inputs['total_s']} "
        f"R{conv_inputs['total_r']} - "
        f"[O] N{outer_n} C{outer_c} M{outer_m} P{outer_p} Q{outer_q} S{outer_s} R{outer_r} - "
        f"[I] N{inner_n} C{inner_c} M{inner_m} P{inner_p} Q{inner_q} S{inner_s} R{inner_r}"
    )


class ONNXIMEstimator(RooflineEstimator):
    frequency = ConfigOption(float, description="Clock frequency in Hz", default=1e9)
    binary_path = ConfigOption(str, description="Path or name of ONNXim binary", default="Simulator")
    engine_path = ConfigOption(str, description="Path to ONNXim root directory", optional=True)
    device_path = ConfigOption(str, description="Path to ONNXim device config JSON")
    output_dir = ConfigOption(str, description="Base output directory for ONNXim artefacts (conv ONNX and model list); resolved relative to cwd if not absolute", default="output/onnxim")
    conv_path = ConfigOption(str, description="Path to conv ONNX output file; overrides output_dir if set", optional=True)
    model_list_path = ConfigOption(str, description="Path to model list JSON output file; overrides output_dir if set", optional=True)

    def __post_init__(self):
        super().__post_init__()
        # Resolve binary: prefer an explicit file path first, then search PATH.
        if os.path.isfile(self.binary_path):
            self.onnxim_binary = os.path.abspath(self.binary_path)
        else:
            self.onnxim_binary = shutil.which(self.binary_path)
            if self.onnxim_binary is None:
                raise FileNotFoundError(
                    f"ONNXim binary '{self.binary_path}' not found as a file or on PATH"
                )

        # Resolve engine root: binary is expected at build/bin/, so root is 2 dirs up.
        if self.engine_path is None:
            binary_dir = os.path.dirname(self.onnxim_binary)
            self.onnxim_path = os.path.normpath(
                os.path.join(binary_dir, os.path.pardir, os.path.pardir)
            )
        else:
            self.onnxim_path = os.path.abspath(self.engine_path)

        engine_configs_dir = os.path.join(self.onnxim_path, "configs")
        if not os.path.isdir(engine_configs_dir):
            raise FileNotFoundError(
                f"ONNXim engine directory '{self.onnxim_path}' does not contain a 'configs' subdirectory"
            )

        self.onnxim_lib_dir = os.path.join(self.onnxim_path, "build", "lib")
        self.device_path = os.path.abspath(self.device_path)

        # Resolve output paths.  Both conv_path and model_list_path can be set
        # individually; if either is absent the shared output_dir provides the
        # base so there is a single anchor to configure.
        base = os.path.abspath(self.output_dir)
        if self.conv_path is None:
            self.conv_path = os.path.join(base, "conv.onnx")
        else:
            self.conv_path = os.path.abspath(self.conv_path)
        if self.model_list_path is None:
            self.model_list_path = os.path.join(base, "model_list.json")
        else:
            self.model_list_path = os.path.abspath(self.model_list_path)

        # Derive all runtime paths that execute_on_onnx will need.
        self.model_name = os.path.splitext(os.path.basename(self.conv_path))[0] or "conv"
        # ONNXIM_HOME is the working tree that ONNXim reads at runtime.  We
        # place it next to the model-list file so all outputs stay together.
        self.onnxim_runtime_home = os.path.join(
            os.path.dirname(self.model_list_path), "onnxim_home"
        )
        self.onnxim_model_dir = os.path.join(self.onnxim_runtime_home, "models", self.model_name)
        self.onnxim_model_path = os.path.join(self.onnxim_model_dir, f"{self.model_name}.onnx")
        self.onnxim_mapping_path = os.path.join(self.onnxim_model_dir, f"{self.model_name}.mapping")
        self.onnxim_runtime_configs_dir = os.path.join(self.onnxim_runtime_home, "configs")
        with open(self.device_path, "r") as f:
            self.onnxim_device_config = json.load(f)

    def get_total_compute_time(self, trace):
        """Extract Total compute time value from log string."""
        match = re.search(r'Total compute time (\d+)', trace)
        return int(match.group(1)) if match else None

    def execute_on_onnx(self, op_info, workload):
        model_json = {
            "models": [
                {
                    "name": self.model_name,
                    "batch_size": 1,
                }
            ]
        }

        os.makedirs(self.onnxim_model_dir, exist_ok=True)
        os.makedirs(self.onnxim_runtime_home, exist_ok=True)
        os.makedirs(os.path.dirname(self.conv_path), exist_ok=True)

        # Symlink the engine configs directory into the runtime home so ONNXim
        # can find them via ONNXIM_HOME without duplicating files.
        if not os.path.exists(self.onnxim_runtime_configs_dir):
            os.symlink(
                os.path.join(self.onnxim_path, "configs"),
                self.onnxim_runtime_configs_dir,
            )

        with open(self.model_list_path, "w") as f:
            json.dump(model_json, f, indent=4)

        onnx.save(workload, self.conv_path)

        mapping_line = create_onnxim_conv_mapping_line(workload, self.onnxim_device_config)
        if mapping_line is None:
            if os.path.exists(self.onnxim_mapping_path):
                os.remove(self.onnxim_mapping_path)
        else:
            with open(self.onnxim_mapping_path, "w") as f:
                f.write(mapping_line)
                f.write("\n")

        # Symlink the model into the expected ONNXIM_HOME location so ONNXim
        # can find it without copying the file.
        if self.conv_path != self.onnxim_model_path:
            if os.path.lexists(self.onnxim_model_path):
                os.remove(self.onnxim_model_path)
            os.symlink(self.conv_path, self.onnxim_model_path)

        cmd = [
            self.onnxim_binary,
            "--config",
            self.device_path,
            "--models_list",
            self.model_list_path,
        ]

        env = os.environ.copy()
        env["ONNXIM_HOME"] = self.onnxim_runtime_home
        if not os.path.isdir(self.onnxim_lib_dir):
            raise FileNotFoundError(
                f"ONNXim library directory '{self.onnxim_lib_dir}' is not a valid directory."
            )

        env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [self.onnxim_lib_dir, env.get("LD_LIBRARY_PATH")]))

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if process.stdout is None:
            raise RuntimeError("ONNXim process did not expose a stdout pipe")

        output_lines = []
        for line in process.stdout:
            output_lines.append(line)
            log.debug(line)

        returncode = process.wait()
        trace = "".join(output_lines)
        if not trace.strip():
            log.error("ONNXim produced no output")
            return OpResult(
                success=False,
                op_info=op_info,
                runtime_estimate=None,
                metadata={},
            )
        if returncode != 0:
            log.error(f"ONNXim failed with return code {returncode}")
            return OpResult(
                success=False,
                op_info=op_info,
                runtime_estimate=None,
                metadata={},
            )

        total_cycles = self.get_total_compute_time(trace)
        if total_cycles is not None:
            runtime = total_cycles / self.frequency
        else:
            runtime = None
        return OpResult(
            success=runtime is not None,
            op_info=op_info,
            runtime_estimate=runtime,
            metadata={},
        )

    @register_op_handler(["stablehlo.convolution", "stablehlo.dot_general"])
    def stablehlo_to_onnxim_initializers(self, op_info, layer_id=0) -> List[Dict[str, Any]]:
        model = stablehlo_ops_to_onnx_model([op_info], model_name=f"conv_layer_{layer_id}")
        model = prepare_onnxim_model(model)
        checker.check_model(model)
        log.info(f"Translated op {op_info.op_name} to ONNX model with {len(model.graph.node)} nodes and {len(model.graph.initializer)} initializers")
        return self.execute_on_onnx(op_info, model)

    @register_op_handler(["stablehlo.concatenate", "stablehlo.reduce_window"])
    def stablehlo_to_onnxim(self, op_info, layer_id=0) -> List[Dict[str, Any]]:
        model = stablehlo_ops_to_onnx_model([op_info], model_name=f"conv_layer_{layer_id}")
        checker.check_model(model)
        log.info(f"Translated op {op_info.op_name} to ONNX model with {len(model.graph.node)} nodes and {len(model.graph.initializer)} initializers")
        return self.execute_on_onnx(op_info, model)
