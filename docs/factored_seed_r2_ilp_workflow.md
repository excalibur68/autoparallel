# Factored Seed + Radius-2 ILP Workflow

This note describes the current exact-solver path that uses a fabric-aware
factored seed to prune the full mesh strategy space, then solves the pruned
radius-2 problem with the normal PuLP/CBC ILP solver.

It is distinct from the faster production experiment path in
`examples/run_meshdisc_seedball_ws512.py`, which uses the same factored seed and
radius-2 pruning but solves the pruned space with TRW-S and lazy costs.

## Scope

Target setting used in the recent LLaMA3-1B experiments:

- Model graph: `autoparallel._testing.models.llama3.Transformer`
- Shape: LLaMA3-1B example settings, usually `dim=2048`, `n_layers=16`,
  `seq_len=8192`
- Mesh: H100-style topology, node size 8
- Input/output constraint: batch sharding on mesh dim 0,
  `(Shard(0), Replicate(), ...)`
- Gradient reduction: higher precision enabled when bf16 params use fp32 reduce
- Seed radius: `2`
- Exact solve: `ShardingOptimizer(..., build_costs=True, build_pulp=True)` plus
  `get_solution()`

## Call Flow

### 1. Trace the model once

The model graph is mesh independent. The benchmark code constructs a seed mesh
only to run AutoParallel tracing:

```python
autop = AutoParallel(
    model,
    input_fn,
    seed_mesh,
    mp_policy=mp_policy,
    repeated_subgraphs=True,
)
autop.build_model_graph()
gm = autop.gm
```

For real `apply_placement` use, run this inside the `AutoParallel` context
manager. For offline benchmarking, the examples call `build_model_graph()`
directly and keep `autop` alive so fake-mode graph state remains valid.

### 2. Build the fabric-aware factored seed

Entry point:

```python
from autoparallel.mesh_search import build_factored_seed

seed = build_factored_seed(
    gm,
    mesh_shape,
    x_sharding,
    cost_model=topo,
    force_grad_reduce_in_higher_precision=True,
    repeated_subgraphs=True,
    one_d_cache=per_worker_cache,
    fabric_aware=True,
)
```

`build_factored_seed` solves one independent 1D exact ILP for each mesh
dimension. For mesh dim `i`:

1. Create a 1D mesh of shape `(mesh_shape[i],)`.
2. Pin input and output to `(x_sharding[i],)`.
3. Add the parameter-memory budget for that 1D mesh.
4. Solve the 1D problem exactly with `ShardingOptimizer.get_solution()`.
5. Store the first output placement chosen for each FX node on that mesh dim.

The per-dim placements are then stacked:

```text
seed[node.name] = (
    placement_chosen_by_dim0_1d_ilp,
    placement_chosen_by_dim1_1d_ilp,
    placement_chosen_by_dim2_1d_ilp,
    ...
)
```

Plain input/output nodes and their grad/tangent nodes are pinned to the
user-provided IO placements so the radius ball cannot exclude the explicit IO
constraints.

### 3. Fabric awareness in the 1D seed solves

The 1D seed solves do not blindly treat every size-8 dimension as node-local.
For NCCL cost models, `build_factored_seed` derives the topology of the original
mesh dimension before launching the 1D ILP:

```python
topo = derive_mesh_dim_topo(full_cost_model, full_mesh_shape, original_dim_idx)
dim_cost_model = replace(full_cost_model, mesh_dim_topo_override=topo)
```

Inside the 1D solve, downstream cost code still sees mesh dim `0`, because the
mesh is `(size,)`. The original dim id has already been encoded in
`mesh_dim_topo_override`, so:

```python
derive_mesh_dim_topo(dim_cost_model, (size,), 0)
```

returns the original full-mesh dim topology. This handles hybrid cases such as
`(16, 8, 4)` dim 1, where the communicator has `n_nodes=4` and `ppn=2`.

The factored-seed cache key includes the derived topology. Same-size 1D solves
only share cache entries when their input placement and fabric topology match.

### 4. Build the radius-2 exact ILP

After the seed is built, create a normal full-mesh `ShardingOptimizer`, but pass
the seed and radius:

```python
from autoparallel.cost_models.collective_runtime_estimation import set_nccl_topo_config
from autoparallel.mesh_search import reset_mesh_search_caches
from autoparallel.optimize_sharding import ShardingOptimizer

set_nccl_topo_config(topo)
reset_mesh_search_caches()

opt = ShardingOptimizer(
    gm,
    mesh,
    force_grad_reduce_in_higher_precision=True,
    repeated_subgraphs=True,
    build_costs=True,
    build_pulp=True,
    strategy_seed=seed,
    strategy_radius=2,
)
opt.add_sharded_input_constraint([x_sharding])
opt.add_sharded_output_constraint([x_sharding])
opt.add_parameter_memory_constraint(0.0, 1.0 / mesh.size())
solution = opt.get_solution()
```

`ShardingOptimizer.__init__` installs the seed into
`autoparallel.shardings.propagation_rules`. During strategy generation, each node
keeps only first-output placements whose Hamming distance from
`seed[node.name]` is at most 2. The current pruning metric is first output
placement, not full OpSpec distance.

Because this is the exact ILP path, `build_costs=True` computes the full
per-edge cost table for the pruned space, and `build_pulp=True` materializes the
PuLP variables and constraints. `get_solution()` then solves the binary ILP with
CBC and extracts a node-to-`OpSpec` solution.

### 5. Difference from radius-2 TRW-S lazy

The TRW-S lazy path shares the first two stages:

```python
seed = build_factored_seed(...)
opt = ShardingOptimizer(
    gm,
    mesh,
    force_grad_reduce,
    repeated_subgraphs=True,
    build_costs=False,
    strategy_seed=seed,
    strategy_radius=2,
)
ApproximateShardingSolver(opt, **seed_ball_approx_kwargs(ndim)).get_solution()
```

The difference is:

- `build_costs=False` skips full edge-cost materialization.
- No PuLP problem is built.
- TRW-S computes costs lazily only for states it touches.
- The objective is taken from `opt.profile["approximate"]["objective"]`.

The exact ILP path is slower but gives the optimal solution inside the radius-2
seed ball.

## Saved Strategy Artifacts

Use `save_placements()` artifacts for real runs. They are small JSON files with
mesh metadata plus per-node output/input placement strings. `load_placements()`
matches by FX node name and pretty-printed strategy specs, then returns the
`dict[Node, OpSpec]` that can be passed to `AutoParallel.apply_placement()`.

The current complete LLaMA3-1B placement artifacts on disk are:

| Path | Strategy | Environment | Reusable for real run? |
|---|---|---|---|
| `out/meshdisc/llama3_1b_3d_realrun_conda752/llama3_1b_2x2x4_full_placements.json` | Full 3D ILP | `torchtitan_conda_prod:752`, PyTorch `2.14.0a0+git06b761f` | Yes, preferred for MAST real runs that use the same conda package and graph shape. |
| `out/meshdisc/llama3_1b_3d_realrun_conda752/llama3_1b_2x2x4_r2_placements.json` | Factored seed + radius-2 + exact ILP | Same as above | Yes, preferred for MAST real runs. |
| `out/meshdisc/llama3_1b_3d_realrun/llama3_1b_2x2x4_full_placements.json` | Full 3D ILP | Local dev environment at solve time | Only for a matching local graph. Do not use this with the MAST conda graph. |
| `out/meshdisc/llama3_1b_3d_realrun/llama3_1b_2x2x4_r2_placements.json` | Factored seed + radius-2 + exact ILP | Local dev environment at solve time | Only for a matching local graph. Do not use this with the MAST conda graph. |

Both artifact sets use:

- Model: `autoparallel._testing.models.llama3.Transformer`
- Variant: LLaMA3-1B, `n_layers=16`, `seq_len=8192`
- Mesh: `(2, 2, 4)` with dim names `("dp", "shard1", "shard2")`
- World size: `16`, with `gpus_per_node=8`
- Input/output placement: `(Shard(0), Replicate(), Replicate())`
- Parameter memory constraint: `(0.0, 1.0 / mesh.size())`
- `force_grad_reduce_in_higher_precision=True`

The conda-matched artifacts are the right starting point for MAST. They were
generated after the first real-run attempt showed that the local placement JSON
contained a node (`neg_16`) that was not present in the MAST conda graph. The
conda-matched files have `4299` saved nodes and successfully loaded in the MAST
real-run script before the job failed later in runtime/compile.

Solve summary for the conda-matched artifacts:

| Strategy | Status | Objective | Build | Solve | Variables | Saved nodes |
|---|---|---:|---:|---:|---:|---:|
| Full ILP | `Optimal` | `73133.54340816716` | `334.9s` | `985.7s` | `9.16M` | `4299` |
| Factored seed + r=2 + ILP | `Optimal` | `73133.54340816716` | `56.7s` | `84.6s` | `1.49M` | `4299` |

Solve summary for the local dev artifacts:

| Strategy | Status | Objective | Build | Solve | Variables | Saved nodes |
|---|---|---:|---:|---:|---:|---:|
| Full ILP | `Optimal` | `74799.32165976637` | `575.5s` | `808.5s` | `8.47M` | `4363` |
| Factored seed + r=2 + ILP | `Optimal` | `74799.32165976637` | `92.0s` | `80.8s` | `1.39M` | `4363` |

The following files are useful for reporting or debugging, but are not directly
loadable strategies:

| Path | Contents | Why not directly reusable? |
|---|---|---|
| `/tmp/factored_seed_r2_ilp_ws512_l16_s8192_typical.jsonl` | Metrics for exact `factored_seed+r=2+ILP` on WS512 LLaMA3-1B typical 3D meshes. | It does not store node-level strategies. |
| `/tmp/llama3_1b_2x4x8_full_opspec_profile.json` | Metrics and mismatch details for `(2,4,8)` LLaMA3-1B raw seed, radius-2 TRW-S lazy, and full ILP. | It is a profile/comparison file, not a complete solution dump. |
| `/tmp/llama3_1b_2x4x8_full_opspec_profile.md` | Human-readable summary of the JSON above. | It is a report. |
| `out/meshdisc/smoke_dsv3_lp_compare_ws128_16x8.json` | Older DSV3 smoke result with a runner inspection-format strategy list. | It is not for LLaMA3 and is not in `save_placements()` format. |

The exact ILP metrics in `/tmp/factored_seed_r2_ilp_ws512_l16_s8192_typical.jsonl`
cover these larger WS512 meshes:

| Mesh | Objective | Full LP objective | Gap | Total time |
|---|---:|---:|---:|---:|
| `(32,4,4)` | `488601.0759` | `487401.0759` | `0.2462%` | `154.3s` |
| `(16,8,4)` | `561353.5943` | `560104.1542` | `0.2231%` | `142.0s` |
| `(8,8,8)` | `765492.6184` | `764343.6327` | `0.1503%` | `155.1s` |
| `(128,2,2)` | `1484603.8436` | `1482626.0776` | `0.1334%` | `183.3s` |

## Saving a Reusable Strategy

For real runs, prefer AutoParallel's placement serialization instead of the
runner inspection JSON. After solving:

```python
solution = opt.get_solution()
opt.save_placements("out/meshdisc/llama3_ws16_2x2x4_r2_ilp_placements.json")
opt.save("out/meshdisc/llama3_ws16_2x2x4_r2_ilp_optimizer.ap")
```

Use `save_placements()` for a lightweight real-run artifact. Use `save()` when
you want to reopen the full optimizer state for debugging, logs, or diffs.

The JSON `strategy` emitted by `examples/run_meshdisc_llama3_seed_vs_final.py`
is useful for rank reports and manual inspection:

```json
{
  "node": "...",
  "op": "...",
  "target": "...",
  "output_placements": [...],
  "input_placements": [...]
}
```

That format is not currently accepted by `load_placements()`.

## Loading Placements in a Real Run

The real run must use the same model graph, mesh shape, mesh dim names, and IO
constraints as the run that saved the placements. The loader validates
`mesh_shape` and `mesh_dim_names`, then matches every saved node by FX node name
and exact output/input spec strings.

Use the conda-matched files for the current MAST comparison:

```python
FULL_PLACEMENTS = (
    "out/meshdisc/llama3_1b_3d_realrun_conda752/"
    "llama3_1b_2x2x4_full_placements.json"
)
R2_PLACEMENTS = (
    "out/meshdisc/llama3_1b_3d_realrun_conda752/"
    "llama3_1b_2x2x4_r2_placements.json"
)
```

With the current public `AutoParallel` API, entering the context still builds
`autop.sharding_optimizer` once. Use `solver="approx", lazy_costs=True` to make
that automatic build lightweight: it skips PuLP and full edge-cost
materialization, but it still performs unpruned strategy enumeration.

For the radius-2 placement, build the same seed-ball optimizer and load the
placement through it:

```python
with AutoParallel(
    model,
    input_fn,
    mesh,
    mp_policy=mp_policy,
    repeated_subgraphs=True,
    solver="approx",
    lazy_costs=True,
) as autop:
    x_sharding = (Shard(0),) + (Replicate(),) * (mesh.ndim - 1)

    seed = build_factored_seed(
        autop.gm,
        tuple(mesh.shape),
        x_sharding,
        cost_model=topo,
        force_grad_reduce_in_higher_precision=True,
        repeated_subgraphs=True,
        fabric_aware=True,
    )

    r2_opt = ShardingOptimizer(
        autop.gm,
        mesh,
        force_grad_reduce_in_higher_precision=True,
        repeated_subgraphs=True,
        build_costs=False,
        strategy_seed=seed,
        strategy_radius=2,
    )
    r2_opt.add_sharded_input_constraint([x_sharding])
    r2_opt.add_sharded_output_constraint([x_sharding])
    r2_opt.add_parameter_memory_constraint(0.0, 1.0 / mesh.size())

    r2_placement = r2_opt.load_placements(R2_PLACEMENTS)
    parallel_model = autop.apply_placement(r2_placement)
```

`build_costs=False` is fine for loading because `load_placements()` only needs
the strategy space, not PuLP variables or costs. The same seed/radius must still
contain the saved radius-2 placement; otherwise matching will fail.

For the full-ILP placement, do not assume the full solution is inside the
radius-2 seed ball. Load it through an unpruned optimizer. In a real run, the
`AutoParallel` context already built one:

```python
with AutoParallel(
    model,
    input_fn,
    mesh,
    mp_policy=mp_policy,
    repeated_subgraphs=True,
    solver="approx",
    lazy_costs=True,
) as autop:
    full_placement = autop.sharding_optimizer.load_placements(FULL_PLACEMENTS)
    parallel_model = autop.apply_placement(full_placement)
```

If you want both placements in one comparison script, construct two fresh model
instances or two fresh `AutoParallel` contexts. `apply_placement()` mutates the
module by applying DTensor sharding, so do not apply full and radius-2
placements sequentially to the same model object.

For one-off validation, it is also fine to compute and immediately apply the
radius-2 solution in the same process. In that case, build the seed-ball
optimizer with the exact ILP settings:

```python
opt = ShardingOptimizer(
    autop.gm,
    mesh,
    force_grad_reduce_in_higher_precision=True,
    repeated_subgraphs=True,
    build_costs=True,
    build_pulp=True,
    strategy_seed=seed,
    strategy_radius=2,
)
opt.add_sharded_input_constraint([x_sharding])
opt.add_sharded_output_constraint([x_sharding])
opt.add_parameter_memory_constraint(0.0, 1.0 / mesh.size())

solution = opt.get_solution()
opt.save_placements("out/meshdisc/llama3_ws16_2x2x4_r2_ilp_placements.json")
parallel_model = autop.apply_placement(solution)
```

Avoiding the initial unpruned strategy enumeration in the `AutoParallel`
context would require a small API extension that passes `strategy_seed` and
`strategy_radius` into the optimizer created by `AutoParallel.__enter__`, or a
direct `apply_sharding_to_model` integration outside the public context manager.

## Sanity Checks Before Real-Run Testing

Before using a saved placement file in a real run:

1. Check `mesh_shape` and `mesh_dim_names` in the placements JSON.
2. Rebuild the same LLaMA3 graph and IO constraints.
3. Use the same topology model, especially `gpus_per_node=8` for the current
   H100 experiments.
4. Use the same PyTorch/torchtitan conda package when targeting MAST; otherwise
   node names may differ even if the Python source looks unchanged.
5. Load through `opt.load_placements()` and fail fast on any node/spec mismatch.
6. Run a short real distributed smoke before collecting performance numbers.
