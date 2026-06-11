# Public Release Scope

## Kept in the public release

- `src/zeoagent/agent/agent.py`
- `src/zeoagent/cli.py`
- `src/zeoagent/config.py`
- `src/zeoagent/tools/cif_resolver.py`
- `src/zeoagent/tools/zeopp.py`
- `src/zeoagent/tools/iza_match.py`
- `src/zeoagent/tools/ring_tools.py`
- `src/zeoagent/tools/separation_support.py`
- `scripts/generate_frameworks_from_contour.py`

## Kept as manual or placeholder integrations

- `src/zeoagent/tools/hpc_generator.py`
  - manual HPC integration contract
- `src/zeoagent/tools/diffusion_predictor.py`
  - provider placeholder without bundled model assets
- `src/zeoagent/tools/gulp_opt.py`
  - manual reference-optimization step

## Intentionally excluded

- private diffusion models and descriptor datasets
- private literature corpora
- proprietary remote generation workflows
- `src/zeoagent/tools/literature_search.py`
- `src/zeoagent/tools/zeolite_recommender.py`
- `scripts/symmetry_refine_cifs.py`
- `scripts/ring_unique_counts.py`
- internal preview and experimental scripts
