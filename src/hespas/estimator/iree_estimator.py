# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import subprocess
import shutil
from pathlib import Path
import json
import logging

from ..utils.iree import get_iree_input_string
from .compute_estimator import ComputeEstimator
from .check_available import check_programs
from .config_option import ConfigOption
from .result import ModuleResult

log = logging.getLogger(__name__)

class IREEEstimator(ComputeEstimator):

    iree_compile = ConfigOption(str, description="Path to the iree-compile binary", default=shutil.which("iree-compile"))
    iree_benchmark = ConfigOption(str, description="Path to the iree-benchmark-module binary", default=shutil.which("iree-benchmark-module"))
    mlir_function_name = ConfigOption(str, description="main function with the MLIR module to execute", default="main")
    compile = ConfigOption(dict, description="Compile arguments to iree")
    sample = ConfigOption(bool, default=False)
    conditions = check_programs("iree", {"iree-compile": iree_compile.default, "iree-benchmark-module": iree_benchmark.default})

    def get_vmfb_file_path(self, module):
        return Path(module.module_path).with_suffix('.vmfb')

    def get_compile_args(self, module):
        compile_args = [
            'iree-compile',
            f"{module.module_path}",
            f"--iree-hal-target-backends={self.compile['target-backends']}",
            f"-o={self.get_vmfb_file_path(module)}",
        ]
        if "extra-args" in self.hw_config['compile']:
            compile_args = compile_args + self.compile['extra-args']
        return compile_args


    def get_exec_args(self, module):
        device = self.hw_config['compile']['device']
        benchmark_repetitions = 2
        report_only_aggregates = True

        exec_args = [
        "iree-benchmark-module",
        f"--device={device}",
        f"--module={self.get_vmfb_file_path(module)}",
        f"--function={self.mlir_function_name}",
        f"--benchmark_report_aggregates_only={str(report_only_aggregates).lower()}",
        f"--benchmark_format={'json'}"
        ]
        exec_args.extend(self.get_inputs(module))
        exec_args = [arg.strip() for arg in exec_args]

        if "extra-runtime-args" in self.compile:
            exec_args.extend(self.compile['extra-runtime-args'])
        rep = False
        for argument in exec_args:
            if '--benchmark_repetitions' in argument:
                rep = True
                break
        if not rep:
            exec_args.append(f"--benchmark_repetitions={benchmark_repetitions}")
        return exec_args


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

        compile_args = self.get_compile_args(module)
        log.info('Compiling MLIR')
        log.info(" ".join(compile_args))

        rc = subprocess.run(compile_args, capture_output=True, text=True)

        if (rc.returncode != 0):
            log.error(f"Error compiling MLIR. stderr: {rc.stderr}")
            return False
        return True


    def iree_runtime_estimate(self, module):
        """
        Estimates runtime performance of an IREE compiled model.

        Args:
            input_vmfb_file (Path): Path to the compiled IREE VMFB file.
            entry (str): Name of the entry function to invoke.
            inputs (list): List of input tensors to pass to the model.

        Uses:
            - Loads the VMFB file and executes the specified entry function.
            - Measures execution time and potentially other runtime metrics.

        Returns:
            dict: A dictionary containing runtime estimates.
        """

        exec_args = self.get_exec_args(module)

        log.info("Executing MLIR:")
        log.info(" ".join(exec_args))
        rc = subprocess.run(exec_args, capture_output=True, text=True)

        if rc.returncode != 0 or 'UNIMPLEMENTED;' in rc.stderr:
            log.error(f"Error running benchmark {rc.returncode}. stdout: {rc.stdout} stderr: {rc.stderr}")
            return None

        return self._parse_benchmark_output(rc.stdout)


    def get_inputs(self, module):
        inputs = []
        for t in module.input_dims:
            inputs.append(get_iree_input_string(t))
        return inputs

    def _get_whole_module_result(self, module):
        """
        Compile the MLIR module using IREE and execute it on the target CPU
        to obtain a runtime estimate in microseconds.
        :return: Estimated runtime in microseconds or None if execution fails.
        """

        metadata = {}
        runtime = None

        # Compile the MLIR module
        ret = self.generic_compile(module)
        metadata['compile_fail'] = not ret
        if ret:
            result = self.iree_runtime_estimate(module)
        # Execute the compiled module
        if result is not None:
            metadata['iree_execution_result'] = result
            metadata['execution_fail'] = False
            runtime = result['mean']['CPUTime(ms)'] / 1000  # convert to s
        else:
            metadata["execution_fail"] = True

        return ModuleResult(success=result is not None, module=module, runtime_estimate=runtime, metadata=metadata)


    def convert_time(self, value, time_unit):
        """Converts time to milliseconds based on the given unit."""
        unit_multipliers = {
            "s": 1000,  # seconds to milliseconds
            "ms": 1,    # milliseconds (no change)
            "us": 0.001, # microseconds to milliseconds
            "ns": 0.000001 # nanoseconds to milliseconds
        }
        if time_unit not in unit_multipliers.keys():
            raise ValueError(f"Invalid {time_unit}")
        return value * unit_multipliers.get(time_unit)

    def _parse_benchmark_output(self, output):
        """
        Extract runtime estimates from the benchmark output.
        """
        data = json.loads(output)

        # Extract benchmark results
        results = {}
        for entry in data["benchmarks"]:
            aggregate_name = entry["aggregate_name"]
            results[aggregate_name] = {
                "RealTime(ms)": self.convert_time(entry["real_time"],entry["time_unit"]),
                "CPUTime(ms)": self.convert_time(entry["cpu_time"],entry["time_unit"]),
                "ItemsperSecond": entry["items_per_second"]
            }
        return results


if __name__ == '__main__':
    IREEEstimator.main()
