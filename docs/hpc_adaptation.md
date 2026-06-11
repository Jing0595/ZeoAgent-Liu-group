# HPC Adaptation

The public ZeoAgent release does not ship with the proprietary remote generation workflow used in the private project.

## What you should run on your HPC system

Use the public point-cloud generation algorithm together with your own cluster-specific job scripts, scheduler settings, software environment, and file staging logic.

Your remote workflow should:

1. accept a reference framework or input CIF together with generation parameters
2. run your generation pipeline on your own HPC system
3. produce candidate CIF files
4. return those CIF files to a local directory visible to ZeoAgent

## What ZeoAgent expects back

ZeoAgent only requires a local directory of candidate CIF files. Once those files exist, ZeoAgent can continue with:

- ring analysis
- Zeo++ pore and channel analysis
- IZA matching
- downstream screening and reporting

## Minimal handoff contract

- input to your remote workflow: reference framework, input CIF, and generation parameters
- output from your remote workflow: one local directory containing candidate `*.cif` files

Point the `hpc_generation` step to that local candidate directory when rerunning ZeoAgent.
