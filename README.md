# ZeoAgent

**ZeoAgent: Autonomous Design of Zeolite Frameworks through Pore-Topology-Guided Generation and Evaluation**

ZeoAgent is a zeolite-specific agent framework that converts natural-language research objectives into computational workflows for zeolite structure analysis, framework screening, and point-cloud-guided framework design.

The central idea behind ZeoAgent is that zeolite design cannot be handled reliably by language-only generation. Zeolite performance depends on periodic pore topology, ring geometry, channel connectivity, and strict tetrahedral framework constraints. In this public release, ZeoAgent combines LLM-based planning with zeolite-specific analysis tools and a public point-cloud generation algorithm so that user requests can be translated into explicit computational operations and auditable intermediate results.

## What this repository contains

- the ZeoAgent orchestration and CLI entrypoints
- a public point-cloud-based framework generation script
- downstream screening tools for ring analysis, Zeo++ pore analysis, and IZA matching
- local evidence-retrieval helpers for separation-oriented tasks
- public integration contracts for optional HPC, diffusion-prediction, and reference-optimization steps

## What users should prepare or integrate

To reproduce the full research workflow, users should prepare or connect the following components in their own environments:

- a diffusion-prediction backend and any associated descriptor data if diffusion prediction is needed
- an HPC-side generation workflow for large-scale candidate generation
- any local literature or separation corpora used for evidence retrieval
- cluster-specific deployment details such as scheduler settings, remote paths, and environment setup
- a local reference-optimization workflow if GULP-based reference relaxation is part of your use case

## Scientific scope

ZeoAgent is designed for workflows where a user expresses a zeolite objective in natural language and the system converts that objective into a structured sequence of tool-grounded operations. In the associated manuscript, this includes representative tasks such as:

- pore and ring analysis of existing zeolite frameworks
- diffusion-related analysis through a pluggable predictor interface
- framework design with structural constraints
- screening for diffusion-relevant pore features
- screening for pore dimensions relevant to CO2/CH4 separation

The public release focuses on the parts of that workflow that can be shared cleanly: orchestration, public analysis tools, and the point-cloud-guided generation algorithm.

## Point-cloud-guided generation

The main public generation component is:

- `scripts/generate_frameworks_from_contour.py`

This script implements the core point-cloud-guided idea described in the manuscript:

1. represent the pore architecture of a reference framework as a point-cloud contour
2. place symmetry-unique T sites outside that contour
3. expand those sites through symmetry operations
4. screen candidate arrangements with geometric and connectivity constraints

This repository provides the algorithm itself. Users are expected to deploy and run that generation workflow in their own HPC environments when large-scale generation is required.

## Public workflow boundary

The open-source ZeoAgent release does **not** automate remote job submission or distribute the proprietary remote generation workflow from the private project.

Instead, the intended handoff is:

1. use the public point-cloud generation algorithm in your own local or HPC environment
2. produce candidate CIF files
3. return those candidate CIF files to a local directory visible to ZeoAgent
4. let ZeoAgent continue with downstream screening using ring analysis, Zeo++, IZA matching, and related checks

See [docs/hpc_adaptation.md](docs/hpc_adaptation.md) for the expected contract.

## Repository layout

- `src/zeoagent/`
  - ZeoAgent framework, CLI, and public tool wrappers
- `scripts/generate_frameworks_from_contour.py`
  - public point-cloud-based framework generation algorithm
- `docs/hpc_adaptation.md`
  - how to connect your own HPC workflow
- `docs/model_integration.md`
  - how to connect your own diffusion predictor
- `docs/data_policy.md`
  - what is intentionally excluded from the public release
- `docs/release_scope.md`
  - module-level public-release boundary

## Tool status in the public release

- `hpc_generation`
  - manual integration contract; users provide candidate CIF outputs from their own HPC workflow
- `diffusion_predictor`
  - provider placeholder; users supply their own model backend and descriptors
- `gulp_opt`
  - manual reference-optimization step; the private force-field setup is not distributed

## Installation

```bash
pip install -r requirements.txt
```

Prepare your own data under:

- `data/cif_files/`
- `data/iza_cif/`
- `data/separation_corpus/` if you want local separation evidence retrieval

Example helper files are provided in:

- `examples/configs/separation_corpus.jsonl.example`
- `examples/configs/molecular_diameters.json`

## CLI usage

```bash
PYTHONPATH=src python -m zeoagent.cli "What is the pore diameter of CHA?"
PYTHONPATH=src python -m zeoagent.cli "Which framework has the larger surface area, CHA or EAB?"
PYTHONPATH=src python -m zeoagent.cli "Generate a new zeolite candidate inspired by CHA"
```

## Design philosophy

This repository intentionally favors explicit tool boundaries, auditable outputs, and user-controlled deployment over hidden automation. The public release is meant to expose the scientific logic of ZeoAgent while keeping private data assets and private infrastructure details out of scope.
