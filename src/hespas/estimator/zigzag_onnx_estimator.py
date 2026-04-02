# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
from importlib import resources
from datetime import datetime
from pathlib import Path
from typing import Any
import tempfile
import os
from functools import lru_cache
from .check_available import EstimatorNotAvailableError
try:
    import onnx
    #from zigzag.api import get_hardware_performance_zigzag
    from ..stablehlo_to_onnx.translator_api import stablehlo_ops_to_onnx_model
    from zigzag.mapping.temporal_mapping import TemporalMappingType
    from zigzag.stages.evaluation.cost_model_evaluation import CostModelStage
    from zigzag.stages.main import MainStage
    from zigzag.stages.mapping.spatial_mapping_generation import SpatialMappingGeneratorStage
    from zigzag.stages.mapping.temporal_mapping_generator_stage import TemporalMappingGeneratorStage
    from zigzag.stages.parser.accelerator_parser import AcceleratorParserStage
    from zigzag.stages.parser.onnx_model_parser import ONNXModelParserStage
    from zigzag.stages.results.reduce_stages import MinimalLatencyStage, SumStage
    from zigzag.stages.workload_iterator import WorkloadStage
except ImportError as e:
    raise EstimatorNotAvailableError("Unable to import zigzag: '{}'".format(e), "zigzag") from e
from ..utils.num_tools import pos_int
from .estimator import ConfigOption, register_op_handler
from .result import OpResult
from .roofline_estimator import RooflineEstimator

_INLINE_ZIGZAG_MAPPINGS = {
    "tpuv3_like_conv.yaml": """- name: Conv
  spatial_mapping:
    D1:
      - K, 64
    D2:
      - C, 2
  memory_operand_links:
    O: O
    W: I2
    I: I1

- name: default
  spatial_mapping:
    D1:
      - K, 64
    D2:
      - C, 2
  memory_operand_links:
    O: O
    W: I2
    I: I1
""",
}

def _get_hardware_performance_zigzag(workload, accelerator, mapping, dump_folder):
    temporal_mapping_type = "uneven"
    pickle_filename = f"{dump_folder}/list_of_cmes.pickle"
    lpf_limit = 6
    nb_spatial_mappings_generated = 3

    stages = [
        ONNXModelParserStage,
        AcceleratorParserStage,
        SumStage,
        WorkloadStage,
        SpatialMappingGeneratorStage,
        MinimalLatencyStage,
        TemporalMappingGeneratorStage,
        CostModelStage
    ]

    mainstage = MainStage(
        list_of_callables=stages,
        accelerator=accelerator,
        workload=workload,
        mapping=mapping,
        dump_folder=dump_folder,
        pickle_filename=pickle_filename,
        loma_lpf_limit=lpf_limit,
        loma_show_progress_bar=False,
        nb_mappings_generated=nb_spatial_mappings_generated,
        enable_mix_spatial_mapping_generation=False,
        access_same_data_considered_as_no_access=True,
        temporal_mapping_type=TemporalMappingType(temporal_mapping_type)
    )

    cmes = mainstage.run()
    energy_total = cmes[0][0].energy_total
    latency_total = cmes[0][0].latency_total2

    return energy_total, latency_total, cmes

get_hardware_performance_zigzag = _get_hardware_performance_zigzag

@lru_cache(maxsize=None)
def _materialize_inline_zigzag_mapping(name: str) -> str | None:
    content = _INLINE_ZIGZAG_MAPPINGS.get(name)
    if content is None:
        return None
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f"zigzag_mapping_{name.replace('.yaml', '')}_",
        suffix=".yaml",
        delete=False,
    ) as f:
        f.write(content)
        return f.name


def _resolve_local_zigzag_mapping(name: str) -> str | None:
    candidate = Path(name)
    if candidate.is_file():
        return str(candidate.resolve())
    return _materialize_inline_zigzag_mapping(name)

class ZigzagONNXEstimator(RooflineEstimator):

    @staticmethod
    def get_zigzag_mapping_path(name):
        local_mapping = _resolve_local_zigzag_mapping(name)
        if local_mapping is not None:
            return local_mapping
        return str(resources.files("zigzag").joinpath("inputs", "mapping", name))

    @staticmethod
    def get_zigzag_accel_path(name):
        return str(resources.files("zigzag").joinpath("inputs", "hardware", name))

    mapping = ConfigOption(str, default="tpuv3_like.yaml", post_validate_func=get_zigzag_mapping_path, description="Mapping configuration file to use")
    accelerator = ConfigOption(str, default="tpuv3_like.yaml", post_validate_func=get_zigzag_accel_path, description="Accelerator configuration file to use")
    frequency = ConfigOption(pos_int, default=1e9, description="Frequency in Hz")


    def translate_mlir_to_onnx_model(self, op_info: Any) -> "onnx.ModelProto":
        """Translate MLIR (or MLIRModule) to an ONNX-IR model."""
        return stablehlo_ops_to_onnx_model(
            [op_info],
        )

    def __post_init__(self):
        super().__post_init__()
        self.mapping = self._normalize_mapping_for_onnx(self.mapping)

    @staticmethod
    def _normalize_mapping_for_onnx(mapping_path: str) -> str:
        """
        ZigZag's ONNX Gemm/MatMul parser uses loop dims C/D/K.
        Most provided mappings use M/N (and implicitly K). Translate M->D, N->K
        in a temp mapping file to avoid per-workload overfitting.
        """
        if os.path.exists(mapping_path):
            with open(mapping_path, "r") as f:
                content = f.read()
        else:
            raise FileNotFoundError(f"ZigZag mapping file not found: {mapping_path}")

        if "M," not in content and "N," not in content:
            return mapping_path

        normalized = content.replace("M,", "D,").replace("N,", "K,")

        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="zigzag_onnx_mapping_",
            suffix=".yaml",
            delete=False,
        ) as f:
            f.write(normalized)
            return f.name

    @register_op_handler(['stablehlo.dot_general', 'stablehlo.convolution'])
    def handle_op(self, op_info):

        workload = self.translate_mlir_to_onnx_model(op_info)
        _, latency, _ = get_hardware_performance_zigzag(
            workload=workload,
            accelerator=self.accelerator,
            mapping=self.mapping,
            dump_folder=f"output/zigzag_dump/{datetime.now()}",
        )
        runtime = float(latency / self.frequency)  # convert to seconds
        return OpResult(success=True, op_info=op_info, runtime_estimate=runtime, metadata={})

if __name__ == '__main__':
    ZigzagONNXEstimator.main()
