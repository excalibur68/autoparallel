# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""3D world_size=512 mesh discovery with the fast solver (seed-ball + TRW-S + lazy
build), parallelized across CPU cores.

Everything is a fake-process-group dry run (shape inference + cost model only), so
no GPU is needed and candidates are embarrassingly parallel: one worker process per
core, each traces the model once and then evaluates the 3D mesh candidates assigned
to it. Per candidate we build a lazy (build_costs=False) seed-ball-pruned
ShardingOptimizer and solve it with the TRW-S ApproximateShardingSolver.
"""

import argparse
import json
import multiprocessing as mp
import time

import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel._testing.models.llama3 import Transformer, TransformerModelArgs
from autoparallel.api import AutoParallel
from autoparallel.approximate_sharding import ApproximateShardingSolver
from autoparallel.cost_models.collective_runtime_estimation import set_nccl_topo_config
from autoparallel.cost_models.nccl_cost_model import h100_topo_config
from autoparallel.mesh_search import reset_mesh_search_caches
from autoparallel.optimize_sharding import ShardingOptimizer

# Set by the parent in main() and read by spawned workers via the initializer args.
_G = {}


def factor_3d(n):
    """All ordered 3D power-of-2 factorizations (d0, d1, d2), product == n, each >= 2.
    d2 is innermost (node-local under the rightmost-is-innermost convention)."""
    out = []
    a = [1 << i for i in range(1, n.bit_length()) if (1 << i) <= n]
    for d0 in a:
        if n % d0:
            continue
        r = n // d0
        for d1 in a:
            if d1 <= r and r % d1 == 0 and (r // d1) >= 2:
                out.append((d0, d1, r // d1))
    return out


def model_args(cfg):
    return TransformerModelArgs(
        dim=2048, n_layers=cfg["n_layers"], n_heads=32, n_kv_heads=8,
        ffn_dim_multiplier=1.5, multiple_of=256, rope_theta=500000,
        vocab_size=128256, max_seq_len=cfg["seq"],
    )


def worker_init(cfg):
    """Per-process: init a fake PG, trace the model once, cache the graph + topo."""
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "fake", store=FakeStore(), rank=0, world_size=cfg["world"]
        )
    margs = model_args(cfg)

    def input_fn():
        return torch.randint(0, margs.vocab_size, (cfg["world"], cfg["seq"]), device="cuda")

    with torch.device("meta"):
        model = Transformer(margs)
        model.output.weight = model.tok_embeddings.weight
    with unset_fake_temporarily():
        seed_mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda", (cfg["world"],), mesh_dim_names=("dp",)
        )
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    autop = AutoParallel(
        model, input_fn, seed_mesh, mp_policy=mp_policy, repeated_subgraphs=True
    )
    autop.build_model_graph()
    _G["gm"] = autop.gm
    _G["autop"] = autop  # keep alive so the fake mode / graph stays valid
    _G["topo"] = h100_topo_config(
        num_nodes=cfg["world"] // cfg["gpn"], gpus_per_node=cfg["gpn"]
    )
    _G["radius"] = cfg["radius"]
    # bf16 params + fp32 grad reduction => reduce in higher precision.
    _G["force_grad_reduce"] = True


def make_seed_builder(ndim, x_sharding):
    """A callable(graph)->{node.name: placements} 'rep' seed: every node centered at
    full Replicate, with plain input/output (+ grad/tangent) nodes pinned to the IO
    constraint placement so the seed-ball never excludes the constrained placement."""

    def build_seed(graph):
        from torch._functorch._aot_autograd.fx_utils import (
            get_plain_input_and_grad_nodes,
            get_plain_output_and_tangent_nodes,
        )

        rep = (Replicate(),) * ndim
        seed = {n.name: rep for n in graph.nodes if n.op != "output"}
        for _desc, (node, comp) in get_plain_input_and_grad_nodes(graph).items():
            seed[node.name] = x_sharding
            if comp is not None:
                seed[comp.name] = x_sharding
        for _desc, (node, comp) in get_plain_output_and_tangent_nodes(graph).items():
            seed[node.name] = x_sharding
            if comp is not None:
                seed[comp.name] = x_sharding
        return seed

    return build_seed


def eval_candidate(shape):
    """Build a lazy seed-ball ShardingOptimizer for `shape` and solve with TRW-S."""
    gm = _G["gm"]
    ndim = len(shape)
    names = ("dp", "cp", "tp") if ndim == 3 else tuple(f"d{i}" for i in range(ndim))
    x_sharding = (Shard(0),) + (Replicate(),) * (ndim - 1)
    try:
        with unset_fake_temporarily():
            mesh = torch.distributed.device_mesh.init_device_mesh(
                "cuda", shape, mesh_dim_names=names
            )
        set_nccl_topo_config(_G["topo"])
        reset_mesh_search_caches()

        t = time.perf_counter()
        opt = ShardingOptimizer(
            gm, mesh, _G["force_grad_reduce"], repeated_subgraphs=True,
            build_costs=False,
            strategy_seed=make_seed_builder(ndim, x_sharding),
            strategy_radius=_G["radius"],
        )
        opt.add_sharded_input_constraint([x_sharding])
        opt.add_sharded_output_constraint([x_sharding])
        opt.add_parameter_memory_constraint(0.0, 1.0 / mesh.size())
        build_s = time.perf_counter() - t

        t = time.perf_counter()
        ApproximateShardingSolver(opt).get_solution()
        solve_s = time.perf_counter() - t

        n_strats = sum(
            len(s.strategies) for s in opt.strats.values() if hasattr(s, "strategies")
        )
        return {
            "shape": list(shape), "feasible": True,
            "objective": float(opt.profile["approximate"]["objective"]),
            "build_s": build_s, "solve_s": solve_s, "n_strats": n_strats,
        }
    except Exception as exc:  # noqa: BLE001 - report per-candidate failures, keep going
        return {"shape": list(shape), "feasible": False, "error": f"{type(exc).__name__}: {exc}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world-size", type=int, default=512)
    ap.add_argument("--gpus-per-node", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--output-json", default="/tmp/meshdisc_ws512_3d.json")
    args = ap.parse_args()

    cfg = {
        "world": args.world_size, "gpn": args.gpus_per_node, "seq": args.seq_len,
        "n_layers": args.n_layers, "radius": args.radius,
    }
    cands = factor_3d(args.world_size)
    workers = min(args.workers, len(cands))
    print(
        f"3D mesh discovery: world_size={args.world_size} candidates={len(cands)} "
        f"workers={workers} model=llama3(dim2048,n_layers={args.n_layers}) "
        f"seq={args.seq_len} solver=seedball-r{args.radius}+TRWS+lazy"
    )

    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=worker_init, initargs=(cfg,)) as pool:
        results = pool.map(eval_candidate, cands, chunksize=1)
    wall = time.perf_counter() - t0

    feasible = sorted(
        (r for r in results if r["feasible"]), key=lambda r: r["objective"]
    )
    infeasible = [r for r in results if not r["feasible"]]

    print(f"\n=== results (wall={wall:.1f}s, {len(feasible)}/{len(cands)} feasible) ===")
    print(f"{'mesh':>16} {'objective':>14} {'build_s':>8} {'solve_s':>8} {'n_strats':>9}")
    for r in feasible:
        print(
            f"{str(tuple(r['shape'])):>16} {r['objective']:>14.1f} "
            f"{r['build_s']:>8.1f} {r['solve_s']:>8.1f} {r['n_strats']:>9}"
        )
    for r in infeasible:
        print(f"{str(tuple(r['shape'])):>16} INFEASIBLE {r['error']}")

    if feasible:
        best = feasible[0]
        print(f"\nBEST 3D mesh: {tuple(best['shape'])}  objective={best['objective']:.1f}")

    with open(args.output_json, "w") as f:
        json.dump({"config": cfg, "wall_s": wall, "results": results}, f, indent=2)
    print(f"saved {args.output_json}")


if __name__ == "__main__":
    main()
