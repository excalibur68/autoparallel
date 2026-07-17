#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Compare raw seed ranking vs seed-ball TRW-S ranking for LLaMA3 meshes."""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import queue
import time
from typing import Any

import torch
from meshdisc_rank_utils import (
    attach_ranks,
    n_strats,
    parse_shape,
    parse_world_sizes,
    print_summary,
    print_table,
    solution_json,
    write_json,
    write_rank_report,
    write_summary_csv,
)
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel._testing.models.llama3 import Transformer, TransformerModelArgs
from autoparallel.api import AutoParallel
from autoparallel.approximate_sharding import ApproximateShardingSolver
from autoparallel.cost_models.collective_runtime_estimation import set_nccl_topo_config
from autoparallel.cost_models.nccl_cost_model import h100_topo_config
from autoparallel.mesh_search import (
    build_factored_seed,
    fabric_exponents_from_nccl_topo,
    generate_fabric_mesh_shapes,
    reset_mesh_search_caches,
    seed_ball_approx_kwargs,
)
from autoparallel.optimize_sharding import ShardingOptimizer


_G: dict[str, Any] = {}


def _model_args(cfg: dict[str, Any]) -> TransformerModelArgs:
    return TransformerModelArgs(
        dim=2048,
        n_layers=cfg["n_layers"],
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.5,
        multiple_of=256,
        rope_theta=500000,
        vocab_size=128256,
        max_seq_len=cfg["seq"],
    )


def _mesh_dim_names(ndim: int) -> tuple[str, ...]:
    if ndim == 1:
        return ("dp",)
    if ndim == 2:
        return ("dp", "tp")
    if ndim == 3:
        return ("dp", "cp", "tp")
    if ndim == 4:
        return ("dp", "ep", "cp", "tp")
    return tuple(f"d{i}" for i in range(ndim))


def _build_optimizer(
    shape: tuple[int, ...],
    seed: dict[str, tuple[Any, ...]],
    radius: int,
    x_sharding: tuple[Any, ...],
) -> ShardingOptimizer:
    with unset_fake_temporarily():
        mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda", shape, mesh_dim_names=_mesh_dim_names(len(shape))
        )
    set_nccl_topo_config(_G["topo"])
    reset_mesh_search_caches()
    opt = ShardingOptimizer(
        _G["gm"],
        mesh,
        _G["force_grad_reduce"],
        repeated_subgraphs=True,
        build_costs=False,
        strategy_seed=seed,
        strategy_radius=radius,
    )
    opt.add_sharded_input_constraint([x_sharding])
    opt.add_sharded_output_constraint([x_sharding])
    opt.add_parameter_memory_constraint(0.0, 1.0 / mesh.size())
    return opt


def _solve_seed_ball(
    shape: tuple[int, ...],
    seed: dict[str, tuple[Any, ...]],
    radius: int,
    x_sharding: tuple[Any, ...],
    max_time_s: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    opt = _build_optimizer(shape, seed, radius, x_sharding)
    build_s = time.perf_counter() - t0
    kwargs = seed_ball_approx_kwargs(len(shape), max_time_s=max_time_s)
    t1 = time.perf_counter()
    solution = ApproximateShardingSolver(opt, **kwargs).get_solution()
    solve_s = time.perf_counter() - t1
    objective = float(opt.profile["approximate"]["objective"])
    payload = {
        "feasible": math.isfinite(objective),
        "objective": objective,
        "build_s": build_s,
        "solve_s": solve_s,
        "n_strats": n_strats(opt),
        "strategy": solution_json(solution, _G["gm"]),
    }
    if not payload["feasible"]:
        payload["error"] = "non-finite objective"
    return payload


def _worker_init(cfg: dict[str, Any]) -> None:
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "fake", store=FakeStore(), rank=0, world_size=cfg["world"]
        )
    margs = _model_args(cfg)

    def input_fn():
        return torch.randint(
            0, margs.vocab_size, (cfg["world"], cfg["seq"]), device="cuda"
        )

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
    _G["autop"] = autop
    _G["topo"] = h100_topo_config(
        num_nodes=cfg["world"] // cfg["gpn"], gpus_per_node=cfg["gpn"]
    )
    _G["force_grad_reduce"] = True
    _G["seed_cache"] = {}
    _G["radius"] = cfg["radius"]
    _G["max_time_s"] = cfg["max_time_s"]
    _G["raw_queue"] = cfg.get("raw_queue")


def _eval_candidate(shape: tuple[int, ...]) -> dict[str, Any]:
    ndim = len(shape)
    x_sharding = (Shard(0),) + (Replicate(),) * (ndim - 1)
    result: dict[str, Any] = {
        "shape": list(shape),
        "mesh_dim_names": list(_mesh_dim_names(ndim)),
    }
    try:
        t0 = time.perf_counter()
        seed = build_factored_seed(
            _G["gm"],
            shape,
            x_sharding,
            cost_model=_G["topo"],
            force_grad_reduce_in_higher_precision=_G["force_grad_reduce"],
            repeated_subgraphs=True,
            one_d_cache=_G["seed_cache"],
            fabric_aware=True,
        )
        seed_s = time.perf_counter() - t0
        result["seed_s"] = seed_s
    except Exception as exc:  # noqa: BLE001
        result["seed_error"] = f"{type(exc).__name__}: {exc}"
        if _G.get("raw_queue") is not None:
            _G["raw_queue"].put(
                {
                    "shape": list(shape),
                    "feasible": False,
                    "error": result["seed_error"],
                    "phase": "seed",
                }
            )
        return result

    try:
        raw_score = _solve_seed_ball(
            shape,
            seed,
            0,
            x_sharding,
            max_time_s=_G["max_time_s"],
        )
        result["raw_seed"] = raw_score
    except Exception as exc:  # noqa: BLE001
        raw_score = {"feasible": False, "error": f"{type(exc).__name__}: {exc}"}
        result["raw_seed"] = raw_score

    if _G.get("raw_queue") is not None:
        _G["raw_queue"].put(
            {
                "shape": list(shape),
                "feasible": bool(raw_score.get("feasible")),
                "objective": raw_score.get("objective"),
                "build_s": raw_score.get("build_s"),
                "solve_s": raw_score.get("solve_s"),
                "n_strats": raw_score.get("n_strats"),
                "phase": "raw_seed",
            }
        )

    try:
        final_score = _solve_seed_ball(
            shape,
            seed,
            _G["radius"],
            x_sharding,
            max_time_s=_G["max_time_s"],
        )
        final_score["seed_s"] = seed_s
        result["final"] = final_score
    except Exception as exc:  # noqa: BLE001
        result["final"] = {"feasible": False, "error": f"{type(exc).__name__}: {exc}"}
    return result


def _fabric_exponents(world_size: int, gpus_per_node: int) -> tuple[tuple[str, int], ...]:
    topo = h100_topo_config(
        num_nodes=max(1, world_size // gpus_per_node),
        gpus_per_node=gpus_per_node,
    )
    return fabric_exponents_from_nccl_topo(world_size, topo)


def _run_world_size(
    args: argparse.Namespace,
    world_size: int,
    shapes: list[tuple[int, ...]],
    ctx: mp.context.BaseContext,
) -> dict[str, Any]:
    cfg = {
        "world": world_size,
        "gpn": args.gpus_per_node,
        "seq": args.seq_len,
        "n_layers": args.n_layers,
        "radius": args.radius,
        "max_time_s": args.max_time_s,
    }
    workers = min(args.workers, len(shapes))
    fabrics = _fabric_exponents(world_size, args.gpus_per_node)
    print(
        f"\nWS{world_size} seed-vs-final: candidates={len(shapes)} "
        f"workers={workers} fabrics={fabrics} "
        f"model=llama3(dim2048,n_layers={args.n_layers}) seq={args.seq_len} "
        f"raw=seed+r0 final=seed+r{args.radius}+TRWS+lazy",
        flush=True,
    )
    world_t0 = time.perf_counter()
    raw_by_shape: dict[tuple[int, ...], dict[str, Any]] = {}
    raw_rank_ready_s = None
    results: list[dict[str, Any]] = []

    with ctx.Manager() as manager:
        raw_queue = manager.Queue()
        worker_cfg = dict(cfg)
        worker_cfg["raw_queue"] = raw_queue
        with ctx.Pool(
            processes=workers, initializer=_worker_init, initargs=(worker_cfg,)
        ) as pool:
            jobs = [pool.apply_async(_eval_candidate, (shape,)) for shape in shapes]
            done_count = 0
            while done_count < len(jobs):
                while True:
                    try:
                        msg = raw_queue.get_nowait()
                    except queue.Empty:
                        break
                    shape = tuple(msg["shape"])
                    if shape not in raw_by_shape:
                        raw_by_shape[shape] = msg
                        print(
                            f"  raw[{len(raw_by_shape)}/{len(shapes)}] "
                            f"shape={shape} objective={msg.get('objective')}",
                            flush=True,
                        )
                        if len(raw_by_shape) == len(shapes):
                            raw_rank_ready_s = time.perf_counter() - world_t0
                            print(
                                f"  raw rank ready for WS{world_size}: "
                                f"{raw_rank_ready_s:.1f}s",
                                flush=True,
                            )
                new_done = sum(1 for job in jobs if job.ready())
                if new_done > done_count:
                    done_count = new_done
                if done_count < len(jobs):
                    time.sleep(0.5)
            while True:
                try:
                    msg = raw_queue.get_nowait()
                except queue.Empty:
                    break
                raw_by_shape.setdefault(tuple(msg["shape"]), msg)
            if raw_rank_ready_s is None and len(raw_by_shape) == len(shapes):
                raw_rank_ready_s = time.perf_counter() - world_t0
            for idx, job in enumerate(jobs, start=1):
                result = job.get()
                results.append(result)
                raw = result.get("raw_seed", {})
                final = result.get("final", {})
                print(
                    f"  final[{idx}/{len(shapes)}] shape={tuple(result['shape'])} "
                    f"raw={raw.get('objective')} final={final.get('objective')}",
                    flush=True,
                )

    wall_s = time.perf_counter() - world_t0
    if raw_rank_ready_s is None:
        raw_rank_ready_s = float("nan")
    order = {shape: idx for idx, shape in enumerate(shapes)}
    results.sort(key=lambda r: order[tuple(r["shape"])])

    rank_info = attach_ranks(results)
    print_table(f"WS{world_size} raw seed ranking", results, "raw_seed")
    print_table(f"WS{world_size} final seed-ball TRW-S ranking", results, "final")
    print(
        f"\nWS{world_size} Spearman rho={rank_info['spearman']['rho']:.6f} "
        f"n={rank_info['spearman']['n']} raw_rank_ready_s={raw_rank_ready_s:.1f}",
        flush=True,
    )
    if rank_info["best"] is not None:
        best = rank_info["best"]
        print(
            f"WS{world_size} best final mesh: "
            f"{tuple(best['shape'])} final_objective={best['final_objective']:.1f} "
            f"raw_seed_rank={best['raw_seed_rank']}",
            flush=True,
        )

    return {
        "config": cfg,
        "world_size": world_size,
        "num_nodes": max(1, world_size // args.gpus_per_node),
        "fabrics": [[name, exp] for name, exp in fabrics],
        "shapes": [list(s) for s in shapes],
        "wall_s": wall_s,
        "raw_rank_ready_s": raw_rank_ready_s,
        "spearman": rank_info["spearman"],
        "best_final": rank_info["best"],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--world-sizes",
        type=parse_world_sizes,
        default=[8, 16, 32, 64, 128, 256],
        help="Comma-separated power-of-two world sizes.",
    )
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--n-layers", type=int, default=16)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--max-time-s", type=float, default=600.0)
    parser.add_argument(
        "--only-shape",
        type=parse_shape,
        action="append",
        help="Restrict to comma-separated shapes whose product matches each world size.",
    )
    parser.add_argument(
        "--output-json",
        default="out/meshdisc/llama3_seed_vs_final_ws8_256.json",
    )
    parser.add_argument("--output-report", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    if args.gpus_per_node != 8:
        raise ValueError("This runner is scoped to nodesize=8 NVLink + RDMA.")

    world_sizes = list(dict.fromkeys(args.world_sizes))
    ctx = mp.get_context("spawn")
    all_t0 = time.perf_counter()
    world_results = []
    for world_size in world_sizes:
        if world_size < args.gpus_per_node:
            raise ValueError("world sizes must be at least gpus_per_node=8")
        topo = h100_topo_config(
            num_nodes=max(1, world_size // args.gpus_per_node),
            gpus_per_node=args.gpus_per_node,
        )
        fabrics = fabric_exponents_from_nccl_topo(world_size, topo)
        shapes = generate_fabric_mesh_shapes(world_size, fabrics)
        if args.only_shape:
            selected = [s for s in args.only_shape if math.prod(s) == world_size]
            if selected:
                allowed = set(shapes)
                unknown = [shape for shape in selected if shape not in allowed]
                if unknown:
                    raise ValueError(
                        f"--only-shape contains non-canonical shapes for "
                        f"WS{world_size}: {unknown}"
                    )
                shapes = list(dict.fromkeys(selected))
        world_results.append(_run_world_size(args, world_size, shapes, ctx))

    summary = [
        {
            "world_size": wr["world_size"],
            "n_shapes": len(wr["shapes"]),
            "raw_rank_ready_s": wr["raw_rank_ready_s"],
            "wall_s": wr["wall_s"],
            "spearman": wr["spearman"],
            "best_final": wr["best_final"],
        }
        for wr in world_results
    ]
    print_summary(summary)

    payload = {
        "config": {
            "model": "llama3_1b_example",
            "world_sizes": world_sizes,
            "gpus_per_node": args.gpus_per_node,
            "seq": args.seq_len,
            "n_layers": args.n_layers,
            "radius": args.radius,
            "max_time_s": args.max_time_s,
            "workers": args.workers,
        },
        "wall_s": time.perf_counter() - all_t0,
        "summary": summary,
        "world_results": world_results,
    }
    write_json(payload, args.output_json)
    print(f"saved {args.output_json}", flush=True)

    report = args.output_report
    if report is None:
        report = str(args.output_json).removesuffix(".json") + "_report.md"
    write_rank_report(
        payload,
        report,
        title="MeshDisc LLaMA3 Seed vs Final Rank Report",
        full_json_path=args.output_json,
    )
    print(f"saved {report}", flush=True)

    csv_path = args.output_csv
    if csv_path is None:
        csv_path = str(args.output_json).removesuffix(".json") + "_summary.csv"
    write_summary_csv(payload, csv_path)
    print(f"saved {csv_path}", flush=True)


if __name__ == "__main__":
    main()
