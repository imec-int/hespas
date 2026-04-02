Configuration
=============

Hespas uses JSON configuration files to define the input workload, estimator backend,
and output settings. This page describes the configuration format and the available
hardware presets in ``configs/``.

Configuration File Format
-------------------------

A configuration file has three top-level fields:

.. code-block:: json

   {
     "mlir_file": "path/to/workload.mlir",
     "split_strategy": "linear_split",
     "perf_estimator": { ... },
     "output_dir": "output/my_experiment"
   }

``mlir_file``
   Path to the input StableHLO MLIR workload file.

``split_strategy``
   How to partition the MLIR program into compute and communication modules.
   Options: ``"linear_split"`` (default) or ``"individual_split"``.

``output_dir``
   Directory where all output artifacts are written.

``clean`` *(optional)*
   If ``true``, clean the output directory before generating traces.

``block_lim`` *(optional)*
   Block size limit for ``linear_split`` (default: 1024).

``cache_dir`` *(optional)*
   Directory for estimator result caching (default: ``<output_dir>/cache``).

Estimator Configuration
-----------------------

The ``perf_estimator`` object selects the estimator backend and its hardware parameters:

.. code-block:: json

   "perf_estimator": {
       "method": "<estimator_name>",
       "hardware": { ... }
   }

``method``
   The estimator backend to use:

   - ``"roofline"`` — Analytical roofline model (no hardware required)
   - ``"xla"`` — XLA HLO runner profiling (recommended for accuracy, requires NVIDIA GPU + CUDA)
   - ``"iree"`` — IREE compilation and profiling (requires ``pip install ".[iree]"``)

   Additional estimators are available but require external dependencies that are not
   bundled with Hespas: ``"zigzag_onnx"``, ``"cocossim"``, ``"onnxim"``.

``hardware``
   Estimator-specific hardware parameters. The available fields depend on the chosen
   estimator method.

Roofline Estimator
^^^^^^^^^^^^^^^^^^

.. code-block:: json

   "perf_estimator": {
       "method": "roofline",
       "hardware": {
           "peak_flops": 312e12,
           "memory_bandwidth": 1935e9,
           "num_npus": 64,
           "in_memory_only_cache": true
       }
   }

- ``peak_flops`` — Peak FLOPS/s of the target hardware.
- ``memory_bandwidth`` — Peak memory bandwidth in bytes/s.
- ``num_npus`` — Number of devices to generate traces for.
- ``per_datatype_flops`` *(optional)* — Dict mapping datatypes to their peak FLOPS/s
  (e.g. ``{"f16": 312e12, "f32": 19.5e12}``). Overrides ``peak_flops`` per datatype.
- ``in_memory_only_cache`` *(optional)* — Use in-memory cache only, no files.
- ``tdp_W`` *(optional)* — TDP in watts for rough power estimation.
- ``hbm_power_ratio`` *(optional)* — HBM power ratio for power estimation.

XLA Estimator
^^^^^^^^^^^^^

.. code-block:: json

   "perf_estimator": {
       "method": "xla",
       "hardware": {
           "type": "nvidiagpu",
           "num_npus": 4,
           "sample": false,
           "translate": {
               "target-backends": "cuda"
           },
           "in_memory_only_cache": true
       }
   }

- ``type`` — Hardware type (e.g. ``"nvidiagpu"``).
- ``translate.target-backends`` — XLA translation target.
- ``sample`` — Enable sampling mode.

IREE Estimator
^^^^^^^^^^^^^^

.. code-block:: json

   "perf_estimator": {
       "method": "iree",
       "hardware": {
           "num_npus": 64,
           "sample": true,
           "compile": {
               "device": "local-task",
               "target-backends": "llvm-cpu",
               "extra-args": ["--iree-llvmcpu-target-cpu=host"],
               "extra-runtime-args": []
           },
           "in_memory_only_cache": true
       }
   }

- ``compile.device`` — IREE device to use (e.g. ``"local-task"`` for CPU, ``"cuda"`` for GPU).
- ``compile.target-backends`` — IREE compilation target (e.g. ``"llvm-cpu"``, ``"cuda"``).
- ``compile.extra-args`` — Additional IREE compiler flags.
- ``compile.extra-runtime-args`` — Additional IREE runtime flags.
- ``sample`` — Enable sampling mode.

Hardware Presets
----------------

Hespas ships with reference hardware configurations in ``configs/``:

Nodes (``configs/nodes/``)
   Pre-configured hardware parameters for individual accelerators. Each node directory
   contains estimator-specific configuration files:

   - NVIDIA GPUs: A100 (PCIe/SXM, 40/80GB), A40, B200, H100, H200, V100
   - Google TPUs: TPUv3

   Example: ``configs/nodes/A100_SXM_80GB/roofline/config.json`` contains the roofline
   parameters (peak FLOPS per datatype, memory bandwidth) for an A100 SXM 80GB GPU.

Systems (``configs/systems/``)
   Multi-GPU system configurations with `ASTRA-sim <https://astra-sim.github.io/>`_
   network backend files. Each system directory contains:

   - ``astra-sim/system.json`` — ASTRA-sim system configuration (scheduling policy,
     collective implementations)
   - ``astra-sim/network.yml`` — Network topology, bandwidth, latency, and NPU count
   - ``astra-sim/remote_memory.json`` — Remote memory configuration

   These files are consumed by ASTRA-sim when running distributed network simulation
   on the generated Chakra traces.

Using Hardware Presets
^^^^^^^^^^^^^^^^^^^^^^

The preset configs are designed to be used directly with ``hespas_chakra_gen``, with the
workload and output specified via command-line flags:

::

    hespas_chakra_gen configs/nodes/A100_PCIe_40GB/roofline/config.json \
        --mlir_file tests/fixtures/mlir/jax_example.mlir \
        --output output/jax_example_a100

Running with ASTRA-sim
^^^^^^^^^^^^^^^^^^^^^^

After generating Chakra traces with ``hespas_chakra_gen``, you can run ASTRA-sim using
the system configuration files from ``configs/systems/``. ASTRA-sim takes the ``dev.*.et``
trace files and the system/network configuration as input to simulate distributed
communication behavior.

Refer to the `ASTRA-sim documentation <https://astra-sim.github.io/>`_ for installation
and usage instructions. See also the ``experiments/`` directory in this repository for
end-to-end examples.
