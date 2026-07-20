# AutoParallel Documentation

This directory contains newcomer guides, conceptual background, troubleshooting
notes, and deeper explanations of how AutoParallel chooses sharding strategies.

If you're new to the project, use the reading order below.

## Start here

- [Getting Started](getting_started.md)
- [Basic Concepts](basic_concepts.md)
- [API Walkthrough](api_walkthrough.md)

## Troubleshooting and reference

- [Troubleshooting](troubleshooting.md)
- [FAQ](faq.md)

## How AutoParallel works

- [How AutoParallel Chooses a Strategy](how_autoparallel_chooses_a_strategy.md)
- [Adaptive Sharding: Sequence-Parallel vs Column-Parallel](adaptive_sharding.md)

## Advanced usage

- [Using `local_map` for MoE and Custom Communication Patterns](local_map_and_moe.md)
- [Running `local_map` MoE on 3D+ Meshes](local_map_higher_rank_meshes.md)
- [Factored Seed + Radius-2 ILP Workflow](factored_seed_r2_ilp_workflow.md)
- [Saving and Loading Optimizer State](save_load.md)
