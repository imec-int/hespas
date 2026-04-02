# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import subprocess
from pathlib import Path
import shlex
import re
import statistics
import logging

from .config_option import ConfigOption
from .compute_estimator import ComputeEstimator
from .check_available import check_programs
from .result import ModuleResult
from ..utils.logging import get_str_divider

log = logging.getLogger(__name__)

class XLAEstimator(ComputeEstimator):
    xla_translate_path = ConfigOption(str, description="Path to the xla-translate binary", default="xla-translate")
    hlo_runner_main_path = ConfigOption(str, description="Path to the hlo_runner_main binary", default="hlo_runner_main")
    sample = ConfigOption(bool, default=False, optional=True)
    translate = ConfigOption(dict, description="Translate options")
    conditions = check_programs("xla", [{"xla_translate_path": xla_translate_path.default}, {"hlo_runner_main_path": ["which", hlo_runner_main_path.default]}])

    def __get_hlo_file_path(self, module):
        return str(Path(module.module_path).with_suffix('.hlo'))

    def get_mlir_to_hlo_args(self, module):
        """Translates MLIR to HLO using the XLA translation tool.
        """
        return [
            self.xla_translate_path,
            "--stablehlo-to-hlo-text",
            "-o",
            self.__get_hlo_file_path(module),
            str(module.module_path)
        ]

    def get_exec_args(self, module):
        return [
            self.hlo_runner_main_path,
            "--num_repeats=5",
            "--profile_execution=true",
            "--hlo_argument_mode=uninitialized",
            self.__get_hlo_file_path(module)
        ]

    def generic_compile(self, module):
        """
        Compiles an MLIR module string into an IREE VMFB (VM Flatbuffer) file.

        Args:
            module_str (str): The MLIR module representation as a string.
            output_vmfb_file (str): The path to the output VMFB file.

        Uses:
            - Calls `ireec.compile_str` to compile the module.
            - Uses the target backend and extra arguments from `self.hw_config['compile']`.
            - Do not include any trailing or leading white spaces in the configuration json
                for the compilation parameters.
            - Depending on the specific GPU, the "sm" version needs to be changed
        Returns:
            True if the compilation was successful, False otherwise.
        """

        mlir_to_hlo_args = self.get_mlir_to_hlo_args(module)
        log.info('Translating MLIR to HLO:')
        log.info(" ".join(shlex.quote(arg) for arg in mlir_to_hlo_args))
        rc = subprocess.run(mlir_to_hlo_args, capture_output=True, text=True)
        if (rc.returncode != 0):
            log.error(f"Error translating MLIR to HLO. stderr: {rc.stderr}")
            return False
        hlo_file = str(self.__get_hlo_file_path(module))
        if not Path(hlo_file).exists():
            log.error(f"Error: HLO file {hlo_file} was not created.")
            return False
        log.debug(rc.stdout + rc.stderr)
        return True


    def xla_runtime_estimate(self, module):
        """
        Measures runtime performance of an XLA compiled model in seconds.
        :return: Estimated runtime in seconds or None if execution fails.
        """

        args = self.get_exec_args(module)
        log.info(" ".join(shlex.quote(arg) for arg in args))
        rc = subprocess.run(args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if (rc.returncode != 0):
            log.error(f"Error executing XLA runtime estimate. stderr: {rc.stderr}")
            return None

        log.info(get_str_divider())
        # combine because bazel treats all diagnostic output as stderr
        return self._parse_benchmark_output(rc.stdout + rc.stderr)


    def _get_whole_module_result(self, module):
        """
        Compile the MLIR module using IREE and execute it on the target CPU
        to obtain a runtime estimate in microseconds.
        :return: Estimated runtime in microseconds or None if execution fails.
        """
        # Fast path: bypass estimation if specified in hardware config

        metadata = {}
        runtime_estimate = None

        if not self.hw_config.get('bypass'):
            # Compile the MLIR module
            ret = self.generic_compile(module)
            if ret:
            # Execute the compiled module
                runtime_estimate = self.xla_runtime_estimate(module)
        else:
            runtime_estimate = self.hw_config.get('bypass', None)

        if runtime_estimate is not None:
            #runtime_estimate *= 1e6 # convert to microseconds
            #runtime_estimate /= 1e3 # convert to microseconds
            runtime_estimate = float(runtime_estimate)

        return ModuleResult(success=runtime_estimate is not None, module=module, runtime_estimate=runtime_estimate, metadata=metadata)


    def _parse_benchmark_output(self, output):
        """
        Extract runtime estimates from the benchmark output.
        :param output: The output string from the benchmark execution.
        :return: Runtime estimate in seconds or None if parsing fails.
        """
        times = re.findall(r"Execution time,.*duration=([\d.]+)", output) # skipping first run since cold start
        if times:
            # Assuming ns - check this
            return statistics.mean(sorted([(float(t) * 1e-9) for t in times])[:-1]) # Remove longest runtime as likely outlier
        else:
            return None

if __name__ == '__main__':
    XLAEstimator.main()

