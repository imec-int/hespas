Hespas Documentation
====================

Hespas is a distributed ML performance estimation tool that takes a StableHLO workload
representation of an ML model and produces a time estimate for that workload.
It splits the StableHLO programs into compute and communication operators. Compute
performance estimation is done using analytical, simulation, or profiling-based estimators.
It then outputs a `Chakra <https://github.com/imec-int/chakra>`_ execution trace annotated with measured timings that is used as
input to ASTRA-sim for network simulation.

Quick Start
-----------

.. code-block:: bash

   git clone https://github.com/imec-int/hespas.git
   cd hespas
   pip install .

Alternatively, you can use `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   uv sync

Generate a Chakra trace for a sample workload:

.. code-block:: bash

   hespas_chakra_gen tests/fixtures/configs/config_roofline_a100.json

.. toctree::
   :maxdepth: 1
   :caption: Overview
   :hidden:

   overview/why_hespas
   overview/architecture

.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:

   user_guide/installation
   user_guide/config
   user_guide/command_line_tools
   user_guide/estimators

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   tutorials/ispass_2026_workshop

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   _autosummary/hespas
