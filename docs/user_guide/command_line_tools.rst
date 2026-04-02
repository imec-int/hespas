Command Line Tools
==================

Hespas Trace Generator
----------------------

The main entry point for Hespas. It runs the full simulation flow: splitting the MLIR
workload, estimating compute times, and generating `Chakra <https://github.com/imec-int/chakra>`_ execution traces. The compute
estimator backend is specified in the configuration file.

**Command:**
::

    hespas_chakra_gen CONFIG_FILE [OPTIONS]

**Options:**

- ``--clean`` — Clean output directory before generating traces
- ``--mlir_file MLIR_FILE`` — Override the MLIR file specified in the config
- ``--num_npus NUM_NPUS`` — Override the number of NPUs in the config
- ``--split_fn SPLIT_FN`` — Override the split function to use
- ``--block_lim BLOCK_LIM`` — Override block size limit for linear_split
- ``--merge`` / ``--no-merge`` — Override merge setting in individual split
- ``--output OUTPUT`` — Override the output directory in the config
- ``--threads THREADS`` — Number of threads for splitting (0 = single-threaded, -1 = auto)

**Example:**
::

    hespas_chakra_gen tests/fixtures/configs/config_roofline_a100.json

**Output:**

The output directory contains the following files:

Split artifacts
   - ``mini_module_<id>.mlir`` — Individual StableHLO modules split and extracted from the input workload.
     Compute modules contain the operations to be estimated; communication modules contain
     collective operations (e.g. all-reduce).
   - ``mini_module_<id>.json`` — Metadata for each MLIR module, including block type
     (``COMP_NODE`` or ``COMM_NODE``), input/output tensor dimensions, operator count,
     and the parent MLIR file.
   - ``private_functions.mlir`` — Shared helper functions referenced by the mini modules.
   - ``dependency_graph.dot`` — The dependency graph between
     modules in Graphviz DOT format.

Estimation results
   - ``module_stats.json`` — The main results file. Contains per-module estimated runtimes,
     FLOPS counts, memory bytes, hardware utilization metrics (``flops_util``, ``mem_bw_util``),
     and per-operator breakdowns. This is the primary file to inspect when debugging
     estimation accuracy.

Trace files
   - ``dev.<device_id>.et`` — Chakra execution traces, one per simulated device. These are
     the final output, ready to be consumed by ASTRA-sim for network simulation.
   - ``comm_group.json`` — Communication group assignments mapping group IDs to device IDs.

.. note::

   The estimator backend used by ``hespas_chakra_gen`` is determined by the configuration file.
   Estimators can also be run independently — see :doc:`estimators` for details.

Helper Tools
------------

MLIR Splitter
^^^^^^^^^^^^^

Splits MLIR files into smaller modules based on communication and computation operations,
without running estimation. Useful for inspecting the workload structure.

::

    hespas_mlir_splitter CONFIG_FILE [OPTIONS]

Options: ``--split_fn {linear_split,individual_split}``, ``--block_lim BLOCK_LIM``, ``--print_meta``,
``--log-level LOG_LEVEL``

MLIR Analyzer
^^^^^^^^^^^^^

Analyzes MLIR files and provides statistics on StableHLO operator counts.

::

    hespas_mlir_analyzer MLIR_FILE [OPTIONS]

Options: ``-s`` / ``--input_sensitive`` (input-sensitive operator matching),
``-f`` / ``--format FORMAT`` (table/csv output), ``-e`` / ``--expand_func_calls``
(expand function calls in the output)

StableHLO to ONNX Converter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Translates a StableHLO ``.mlir`` file to ONNX format. Useful for interfacing with
ONNX-based estimators and analysis tools.

::

    hespas_stablehlo_to_onnx MLIR_PATH OUT_PATH [OPTIONS]

Options: ``--model-name MODEL_NAME`` (name of the ONNX model, default: ``stablehlo_model``),
``--no-fallback`` (disable fallback for unsupported ops)
