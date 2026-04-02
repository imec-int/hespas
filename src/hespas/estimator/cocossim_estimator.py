# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import logging
import subprocess
import re
import os
import shutil
from typing import List
from .check_available import check_programs
from .estimator import register_op_handler
from .config_option import ConfigOption
from .roofline_estimator import RooflineEstimator
from .result import OpResult
from ..utils.logging import get_str_divider, get_divided_title

log = logging.getLogger(__name__)

class CocossimEstimator(RooflineEstimator):
    cocossim_path = ConfigOption(str, description="Path to cocossim binary", default="perf_model")
    frequency = ConfigOption(float, default=0.94)
    chips = ConfigOption(int, default=1)
    conditions = check_programs("cocossim", [{"cocossim_path": ["which", cocossim_path.default]}])

    def __post_init__(self):
        super().__post_init__()
        self.cocossim_dir = os.path.dirname(shutil.which(self.cocossim_path))

    def stablehlo_to_cocossim_dot_general(self, op_info, layer_id: int = 0) -> List[str]:
        """
        Convert a StableHLO dot_general to a Cocossim-compatible Matmul M K N.
        Only non-contract, non-batch dims form M (LHS) and N (RHS).
        Contract dims form K.
        """

        lhs_shape = list(op_info.input_types[0][0])
        rhs_shape = list(op_info.input_types[1][0])

        dims = getattr(op_info, "dims", {})
        lhs_contract = list(dims.get("lhs", []))
        rhs_contract = list(dims.get("rhs", []))
        batch_pairs = list(dims.get("batch", []))

        # Batch dims only if explicitly paired and same size
        lhs_batch_axes = {p[0] for p in batch_pairs}
        rhs_batch_axes = {p[1] for p in batch_pairs}

        # Contract K
        K_lhs = 1
        for i in lhs_contract:
            K_lhs *= lhs_shape[i]

        K_rhs = 1
        for i in rhs_contract:
            K_rhs *= rhs_shape[i]

        if K_lhs != K_rhs:
            raise ValueError(f"Contracting size mismatch: lhs={K_lhs}, rhs={K_rhs}")

        K = K_lhs

        # M comes from LHS dims not in contract or batch
        M_dims = [lhs_shape[i] for i in range(len(lhs_shape))
                if i not in lhs_contract and i not in lhs_batch_axes]

        M = 1
        for d in M_dims:
            if d != 1:   # ignore leading size-1 dims
                M *= d

        # N comes from RHS dims not in contract or batch
        N_dims = [rhs_shape[i] for i in range(len(rhs_shape))
                if i not in rhs_contract and i not in rhs_batch_axes]

        N = 1
        for d in N_dims:
            if d != 1:
                N *= d

        # hack to avoid overflow bug in cocosim, swap bigger M with N flops wise its the same.
        # even though performance may not be ideal its better than a crash.
        if (M>N):
            tmp = N
            N = M
            M = tmp

        layer = f"Matmul {M} {K} {N}"

        log.info(get_divided_title('Cocossim Workload'))
        log.info(str(layer))
        log.info(get_str_divider())

        return [layer]

    @register_op_handler(['stablehlo.dot_general'])
    def handle_dot_general_cocosim(self, op_info):
        # Build/convert workload
        workload = self.stablehlo_to_cocossim_dot_general(op_info)

        # Write workload to layers.txt
        wfile = os.path.join(self.cocossim_dir, 'layers.txt')
        with open(wfile, "w") as f:
            for line in workload:
                f.write(f"{line}\n")

        # Build the command exactly as required
        cmd = [
            self.cocossim_path,
            "-c", str(self.chips),
            "-sa_sz", "128",
            "-vu_sz", "1024",
            "-ws", "1",
            "-f", str(self.frequency),
            "-i", wfile,
            "-o", 'out.txt'
        ]

        # Run the subprocess
        rc = subprocess.run(cmd, capture_output=True, text=True, cwd=self.cocossim_dir)

        if rc.returncode != 0:
            log.error(f"Error running perf_model. stderr: {rc.stderr}")
            return OpResult(success=False, op_info=op_info, runtime_estimate=None, metadata={})

        log.info(str(rc.stdout))
        match = re.search(r"Time:\s*([0-9.]+)µs", rc.stdout)
        runtime = float(match.group(1)) * 1e-6 if match else None
        return OpResult(success=True, op_info=op_info, runtime_estimate=runtime, metadata={})


    @register_op_handler(["mhlo.fusion"])
    def handle_fusion(self, op_info):
        total_flops = 0
        cocosim_runtime = 0
        total_bytes_accessed = sum([op_info.get_input_bytes(i) for i in range(len(op_info.input_types))]) + sum([op_info.get_output_bytes(i) for i in range(len(op_info.output_types))])
        for fused_op in op_info.fused_ops:
            fused_op_name = fused_op.op_name
            if fused_op_name in self._op_handlers:
                #r, op_flops, _ = self._Estimator__get_op_estimate(fused_op)
                op_result = self._Estimator__get_op_estimate(fused_op)
                if fused_op_name in ['stablehlo.dot_general']:
                    cocosim_runtime += op_result.runtime_estimate
                else:
                    total_flops += op_result.metadata['flops']
            else:
                log.warning("Warning: Unhandled fused operation: {}".format(fused_op_name))

        op_result = self.compute_runtime(op_info, total_flops, total_bytes_accessed)
        # add the cocosim latency
        op_result.runtime_estimate += cocosim_runtime
        return op_result


if __name__ == '__main__':
    CocossimEstimator.main()
