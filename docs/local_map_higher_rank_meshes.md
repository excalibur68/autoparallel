# Running `local_map` MoE on 3D+ Meshes

[Using `local_map` for MoE](local_map_and_moe.md) assumes a fixed 2D
`("dp", "ep")` mesh. [Mesh shape discovery](mesh_shape_search.md), on the other
hand, evaluates the *same* traced graph against many candidate meshes, including
3D and 4D shapes such as `("dp", "ep", "d2", "d3")`. These two features conflict:
`local_map` placements are authored for one specific mesh rank, but discovery
needs them to work at any rank.

This document explains why that conflict exists and how it is resolved by a small
metadata-normalization pass. The reference implementation lives in
`examples/run_meshdisc_dsv3_seed_vs_final.py`; the same approach applies to any
runner that searches mesh shapes over a `local_map` model.

## Why `local_map` is rank-specific

When you write a `local_map` region, every placement entry is a tuple with **one
element per mesh dimension** (see [the MoE guide](local_map_and_moe.md)). The
DeepSeekV3 model authors these for a 2D `("dp", "ep")` mesh:

```python
in_placements=(
    (Shard(0), Shard(0)),       # x:          shard on dp, shard on ep
    (Replicate(), Shard(0)),    # expert_w1:  replicate on dp, one expert per ep rank
    ...
)
out_placements=(
    (Shard(0), Shard(0)),
    (Partial("sum"), Partial("sum")),
)
```

These tuples are stored verbatim on each FX node as
`node.meta["local_map_kwargs"]`, and their length equals the mesh `ndim` at
authoring time — here, 2.

The optimizer consumes them by building a `DTensorSpec` directly from each tuple:

```python
DTensorSpec(mesh=mesh, placements=placement, tensor_meta=tm)
```

`DTensorSpec` requires `len(placements) == mesh.ndim`. So the moment discovery
hands the optimizer a 3D candidate mesh while the metadata still carries 2-tuples,
the spec is malformed. The factored-seed path makes this sharper still: it solves
each mesh dimension **independently as a 1D problem**, so for those sub-solves the
placements must be 1-tuples.

The fix is to rewrite `local_map_kwargs` to the target mesh rank *before* the
graph reaches the optimizer.

## The three rewrite rules

A single normalization pass walks every node carrying `local_map_kwargs` and
rewrites `in_placements`, `out_placements`, and `in_grad_placements`. Three rules
cover every case.

### 1. Expand — pad new axes with `Replicate`

A 2-tuple authored for `("dp", "ep")` is padded on the right to the candidate's
full `ndim`:

```python
expanded = placements + (Replicate(),) * (ndim - len(placements))
```

The extra axes (`d2`, `d3`) are further factors carved from the same physical
fabric. The `local_map` region never declared how to split along them, so the
safe, neutral default is `Replicate()`: the `local_map` tensors do not shard on
the new axes, leaving those axes for the rest of the graph and the ILP to use.

A placement rank that *exceeds* the candidate `ndim` is an error — a candidate
mesh can never have fewer dimensions than the authored placements. (Discovery
already filters candidates to `ndim >= 2`.)

### 2. Degenerate — collapse size-1 axes to `Replicate`

Candidate shapes may contain a size-1 dimension. On such an axis `Shard` and
`Partial` are meaningless (one shard = no shard), so they degenerate to
`Replicate()`:

```python
if mesh_size == 1 and isinstance(placement, (Partial, Shard)):
    placement = Replicate()
```

### 3. Project — select a single axis for factored 1D solves

The factored seed solves one mesh dimension at a time on a 1D mesh. For dimension
`dim_idx`, each placement is first expanded to the full rank, then projected to a
1-tuple:

```python
projected = (expanded[dim_idx],)
```

So `(Shard(0), Shard(0))` projects to `(Shard(0),)` on the `dp` solve, and to
`(Replicate(),)` on a padded `d2` solve.

## Dropping the baked-in mesh

The normalization also sets `device_mesh = None` on every `local_map_kwargs`.
The original value is the concrete 2D mesh captured when the model was built;
leaving it in place would force the optimizer to use that stale mesh instead of
the candidate mesh passed to `ShardingOptimizer`.

## Where the rewrites happen

There are two call sites, and they target different ranks:

1. **Full expansion, once per candidate.** Right after the graph is built, the
   metadata is expanded to the candidate's full `ndim` (rule 1 + rule 2). The
   N-D solves that score the mesh reuse this state directly.

   ```python
   autop.build_model_graph()
   _normalize_local_map_metadata(autop.gm, shape)   # dim_idx=None
   ```

2. **Temporary projection, per factored-seed dimension.** Each 1D sub-solve
   needs a *different* 1-tuple projection (rule 3), and must restore the N-D
   state afterward because later N-D solves reuse the same graph. A context
   manager snapshots the current metadata, projects it, and restores it on exit:

   ```python
   with _local_map_metadata(gm, mesh_shape, dim_idx=dim_idx):
       opt = ShardingOptimizer(gm, mesh_1d, ...)
       solution = opt.get_solution()
   # metadata restored to the full N-D version here
   ```

   The snapshot/restore is what keeps the per-dimension projections from
   polluting one another and preserves the full-rank metadata for the scoring
   solves.

## Summary of the contract

| Situation | Rewrite |
|---|---|
| Candidate mesh has more dims than authored | pad new (rightmost) axes with `Replicate` |
| A mesh axis has size 1 | degenerate `Shard`/`Partial` to `Replicate` |
| Factored 1D per-axis seed solve | project to the single axis's placement |
| Any `local_map` node | clear the baked-in `device_mesh` |

The key idea: `local_map` placements are written against a fixed mesh rank, so
to search arbitrary N-D meshes (and the per-axis 1D sub-problems that seed them)
you normalize the placement metadata to the target rank before optimizing. New
axes default to `Replicate` because the hand-written region never claimed them;
everything else follows from keeping the metadata consistent with whatever mesh
the optimizer is currently solving against.

## Pitfalls

**The rewrite changes placement *rank*, not the number of entries.**
`in_placements` still has one entry per traced input, so the optimizer's
activation/parameter split (`len(user_args) - len(in_placements)`) is unaffected.
Only the per-mesh-dimension tuples grow or shrink.

**Always restore projected metadata.** If a 1D factored solve projects the
metadata and does not restore it, the subsequent N-D scoring solve will see
1-tuples against an N-D mesh and fail. Use the snapshot/restore context manager
rather than mutating `node.meta` in place and hoping to undo it later.

**Padding with `Replicate` is a default, not a decision.** It means "the
`local_map` region does not shard on this new axis." If you *want* the expert
region to exploit a third axis, that requires authoring the placements for that
axis in the model — discovery will not invent a sharding for an opaque region.
