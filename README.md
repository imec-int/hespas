# Hespas

Hespas is a distributed ML performance estimation tool built on [StableHLO](https://openxla.org/stablehlo).
It takes a StableHLO workload representation of an ML model, splits it into compute and
communication regions, estimates compute performance using analytical, simulation, or
profiling-based backends, and outputs [Chakra](https://github.com/imec-int/chakra) execution
traces for network simulation with [ASTRA-sim](https://astra-sim.github.io/).

One workload representation. Multiple simulation backends. Multiple hardware targets.

## Installation

```bash
git clone https://github.com/imec-int/hespas.git
cd hespas
pip install .
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Quick Start

Generate a Chakra trace using the roofline estimator:

```bash
hespas_chakra_gen tests/fixtures/configs/config_roofline_a100.json
```

Use a hardware preset with a custom workload:

```bash
hespas_chakra_gen configs/nodes/A100_PCIe_40GB/roofline/config.json \
    --mlir_file tests/fixtures/mlir/jax_example.mlir \
    --output output/jax_example_a100
```

## Main Estimator Backends

| Backend | Type | Hardware Required |
|---------|------|-------------------|
| **Roofline** | Analytical | None |
| **XLA** | Profiling | NVIDIA GPU + CUDA |
| **IREE** | Compilation + profiling | CPU or GPU |

## CLI Tools

| Command | Description |
|---------|-------------|
| `hespas_chakra_gen` | Main tool — split, estimate, and generate Chakra traces |
| `hespas_mlir_splitter` | Split MLIR into compute/communication modules |
| `hespas_mlir_analyzer` | Analyze StableHLO operator statistics |
| `hespas_stablehlo_to_onnx` | Convert StableHLO to ONNX format |

## Publication

> **Evaluating Cross-Architecture Performance Modeling of Distributed ML Workloads Using StableHLO**
>
> Jonas Svedas, Nathan Laubeuf, Ryan Harvey, Arjun Singh, Changhai Man, Abubakr Nada, Tushar Krishna, James Myers, Debjyoti Bhattacharjee
>
> *IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS), 2026*
>
> [arXiv:2604.12090](https://arxiv.org/abs/2604.12090)

## License

[MIT](LICENSE) — © imec
