Estimators
==========

Hespas supports multiple compute estimator backends. While the main ``hespas_chakra_gen``
tool selects the estimator from the configuration file, each estimator can also be run
independently on a StableHLO MLIR file. This is useful for quick experiments, debugging,
or comparing estimator results without running the full trace generation flow.

All estimators accept a ``--config-file`` for loading parameters from a JSON configuration,
or individual command-line flags to override specific settings. They also support a
``--cache-dir`` for caching results across runs.

Roofline Estimator
------------------

Analytical performance estimator based on the roofline model. Estimates execution time
from peak FLOPS and memory bandwidth — no hardware required.

::

    python -m hespas.estimator.roofline_estimator [OPTIONS]

**Key options:**

- ``--config-file CONFIG_FILE`` — Configuration file path
- ``--mlir-file MLIR_FILE`` — MLIR file to estimate
- ``--peak-flops PEAK_FLOPS`` — Peak FLOPS/s of the target hardware
- ``--memory-bandwidth MEMORY_BANDWIDTH`` — Peak memory bandwidth in bytes/s
- ``--per-datatype-flops PER_DATATYPE_FLOPS`` — Peak FLOPS/s dict for each datatype
- ``--cache-dir CACHE_DIR`` — Cache directory (default: ``hespas_cache``)
- ``--disable-cache`` — Disable the estimator cache
- ``--in-memory-only-cache`` — Use in-memory cache only, no files
- ``--num-npus`` — Number of NPUs to simulate
- ``--error-on-unknown-type`` — Error on unrecognized datatypes
- ``--warn-on-unknown-type`` — Warn on unrecognized datatypes

**Example:**
::

    python -m hespas.estimator.roofline_estimator \
        --config-file tests/fixtures/configs/config_roofline_a100.json

XLA Estimator
-------------

Profiling-based estimator using XLA's HLO runner. Translates StableHLO to HLO and
executes on GPU hardware for measured timings. Use this for the most accurate results.

::

    python -m hespas.estimator.xla_estimator [OPTIONS]

**Key options:**

- ``--config-file CONFIG_FILE`` — Configuration file path
- ``--mlir-file MLIR_FILE`` — MLIR file to estimate
- ``--cache-dir CACHE_DIR`` — Cache directory (default: ``hespas_cache``)
- ``--disable-cache`` — Disable the estimator cache
- ``--in-memory-only-cache`` — Use in-memory cache only, no files
- ``--num-npus`` — Number of NPUs to simulate
- ``--translate TRANSLATE`` — Translation options
- ``--hlo-runner-main-path PATH`` — Path to ``hlo_runner_main`` binary (default: ``hlo_runner_main``)
- ``--xla-translate-path PATH`` — Path to ``xla-translate`` binary (default: ``xla-translate``)
- ``--sample`` — Enable sampling mode

.. note::

   Requires an NVIDIA GPU with CUDA, and the ``xla-translate`` and ``hlo_runner_main``
   binaries (typically built from XLA source or available in a JAX-Toolbox Docker image).

Docker Setup
^^^^^^^^^^^^

The easiest way to get all XLA estimator dependencies is via Docker. The provided
``Dockerfile`` builds an image with ``xla-translate``, ``hlo_runner_main``, and GPU support
pre-configured.

**Build the image:**

.. code-block:: bash

   docker build -t hespas .

**Run with GPU access:**

.. code-block:: bash

   docker run --gpus all -it --rm hespas

**Mount your local workspace:**

.. code-block:: bash

   docker run --gpus all -it --rm -v $(pwd):/workspace -w /workspace hespas

IREE Estimator
--------------

Compilation-and-profiling estimator using the `IREE <https://iree.dev/>`_ compiler and runtime.
Compiles StableHLO modules and benchmarks them on real hardware (CPU or GPU).

Requires the ``iree`` optional dependency:

.. code-block:: bash

   pip install ".[iree]"

::

    python -m hespas.estimator.iree_estimator [OPTIONS]

**Key options:**

- ``--config-file CONFIG_FILE`` — Configuration file path
- ``--mlir-file MLIR_FILE`` — MLIR file to estimate
- ``--cache-dir CACHE_DIR`` — Cache directory (default: ``hespas_cache``)
- ``--disable-cache`` — Disable the estimator cache
- ``--in-memory-only-cache`` — Use in-memory cache only, no files
- ``--num-npus`` — Number of NPUs to simulate

.. note::

   Requires ``iree-compile`` and ``iree-benchmark-module`` binaries to be available on
   the system PATH.
