# ZeoAgent

ZeoAgent is an open framework for zeolite analysis and point-cloud-driven structure-generation workflows.

This public release includes:

- the ZeoAgent orchestration and CLI entrypoints
- the public point-cloud generation algorithm
- downstream screening utilities for ring analysis, Zeo++ metrics, and IZA matching
- local evidence retrieval helpers for separation-oriented tasks

This public release does **not** include:

- private pretrained diffusion models
- private descriptor datasets
- proprietary remote generation workflows
- private literature corpora, analysis outputs, or HPC deployment details

## Public workflow boundary

The point-cloud generation algorithm is provided as source code, but users are expected to deploy and run that workflow on their own HPC systems. Once candidate CIF files are returned locally, ZeoAgent can continue with downstream screening.

## Repository layout

- `src/zeoagent/`: ZeoAgent framework and public tools
- `scripts/generate_frameworks_from_contour.py`: public point-cloud generation algorithm
- `docs/hpc_adaptation.md`: how to connect your own HPC workflow
- `docs/model_integration.md`: how to connect your own diffusion predictor
- `docs/data_policy.md`: what is intentionally excluded from the public release

## Status of selected tools

- `hpc_generation`: manual integration contract; users supply candidate CIF outputs from their own HPC workflow
- `diffusion_predictor`: interface placeholder; users supply their own model backend
- `gulp_opt`: manual integration step; the internal GULP setup is not distributed

## Installation

```bash
pip install -r requirements.txt
```

Prepare your own data under:

- `data/cif_files/`
- `data/iza_cif/`
- `data/separation_corpus/` if you want local separation evidence retrieval

## CLI

```bash
PYTHONPATH=src python -m zeoagent.cli "What is the pore diameter of CHA?"
PYTHONPATH=src python -m zeoagent.cli "Generate a new zeolite candidate inspired by CHA"
```

## Notes

The public repository intentionally favors interface contracts and documentation over private deployment automation. See `docs/` for the expected handoff points.
