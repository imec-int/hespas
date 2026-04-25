Installation
============

Prerequisites
-------------

- Python 3.11 or higher

Installation with pip
---------------------

.. code-block:: bash

   git clone https://github.com/imec-int/hespas.git
   cd hespas
   pip install .

Alternatively, you can use `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   uv sync

Optional dependencies
---------------------

Hespas supports several optional estimator backends. Install them as needed:

.. code-block:: bash

   # IREE compiler and runtime (CPU/GPU execution)
   pip install ".[iree]"

   # ZigZag architectural simulator
   pip install ".[zigzag_onnx]"

   # External StableHLO to ONNX translation package
   pip install ".[stablehlo_to_onnx]"

Running Tests
-------------

.. code-block:: bash

   python -m pytest tests

For verbose output with debug logging:

.. code-block:: bash

   python -m pytest -s --log-cli-level=DEBUG tests

Running a Simple Simulation
----------------------------

Generate a Chakra trace for a sample workload:

.. code-block:: bash

   hespas_chakra_gen tests/fixtures/configs/config_roofline_a100.json

The output will be written to ``output/basic/roofline_a100/`` and includes:

- ``mini_module_*.mlir`` — individual MLIR compute modules
- ``intermediate_results.json`` — execution times and compilation status
- ``dev.*.et`` — `Chakra <https://github.com/imec-int/chakra>`_ trace files for use with ASTRA-sim
