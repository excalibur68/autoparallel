#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Compare DSV3 seed-ball TRW-S lazy results with full LP relaxations."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import torch
from meshdisc_rank_utils import parse_shape, write_json
from run_meshdisc_dsv3_seed_vs_final import (
    _G,
    _build_dsv3_factored_seed,
    _input_sharding,
    _mesh_dim_names,
    _solve_seed_ball,
    _trace_candidate,
)
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel.cost_models.collective_runtime_estimation import set_nccl_topo_config
from autoparallel.cost_models.nccl_cost_model import h100_topo_config
from autoparallel.mesh_search import reset_mesh_search_caches
from autoparallel.optimize_sharding import ShardingOptimizer


DEFAULT_SHAPES = ((16, 8), (2, 8, 8), (4, 4, 8))


def _shape_key(shape: tuple[int, ...]) -> str:
    return ",".join(str(x) for x in shape)


def _load_seedball(path: str | Path | None) -> dict[tuple[int, ...], dict[str, Any]]:
    if path is None or not Path(path).exists():
        return {}
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    out: dict[tuple[int, ...], dict[str, Any]] = {}
    for world in payload.get("world_results", []):
        for row in world.get("results", []):
            shape = tuple(row.get("shape", []))
            final = row.get("final", {})
            if final.get("feasible"):
                out[shape] = {
                    "objective": final.get("objective"),
                    "build_s": final.get("build_s"),
                    "solve_s": final.get("solve_s"),
                    "n_strats": final.get("n_strats"),
                    "source": str(path),
                }
    return out


def _init_worker(cfg: dict[str, Any]) -> None:
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "fake", store=FakeStore(), rank=0, world_size=cfg["world"]
        )
    _G.clear()
    _G.update(cfg)
    _G["topo"] = h100_topo_config(
        num_nodes=cfg["world"] // cfg["gpn"],
        gpus_per_node=cfg["gpn"],
    )
    _G["force_grad_reduce"] = True
    _G["seed_cache"] = {}


def _compute_seedball(
    gm: torch.fx.GraphModule,
    shape: tuple[int, ...],
    max_time_s: float,
) -> dict[str, Any]:
    x_sharding = _input_sharding(shape)
    seed_t0 = time.perf_counter()
    seed, repair_count = _build_dsv3_factored_seed(gm, shape, x_sharding)
    seed_s = time.perf_counter() - seed_t0
    final = _solve_seed_ball(
        gm,
        shape,
        seed,
        _G["radius"],
        x_sharding,
        max_time_s=max_time_s,
    )
    final["seed_s"] = seed_s
    final["seed_repair_changes"] = repair_count
    final["source"] = "recomputed"
    return final


def _solve_full_lp(gm: torch.fx.GraphModule, shape: tuple[int, ...]) -> dict[str, Any]:
    x_sharding = _input_sharding(shape)
    with unset_fake_temporarily():
        mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda", shape, mesh_dim_names=_mesh_dim_names(len(shape))
        )

    set_nccl_topo_config(_G["topo"])
    reset_mesh_search_caches()
    t0 = time.perf_counter()
    opt = ShardingOptimizer(
        gm,
        mesh,
        _G["force_grad_reduce"],
        repeated_subgraphs=True,
        build_costs=True,
        build_pulp=True,
    )
    opt.add_sharded_input_constraint([x_sharding])
    opt.add_sharded_output_constraint([x_sharding])
    opt.add_parameter_memory_constraint(0.0, 1.0 / mesh.size())
    opt._set_objective()
    build_s = time.perf_counter() - t0

    lp = opt.solve_lp_relaxation(verbose=_G["verbose_lp"], extract=False)
    return {
        "feasible": lp["status"] == "Optimal" and lp["objective"] is not None,
        "objective": float(lp["objective"]) if lp["objective"] is not None else None,
        "status": lp["status"],
        "build_s": build_s,
        "solve_s": lp["solve_time"],
        "n_fractional": lp["n_fractional"],
        "n_vars": lp["n_vars"],
        "unique_variables": len(opt.pulp_variables),
        "constraints": len(opt.prob.constraints),
        "n_strats": sum(
            len(s.strategies)
            for s in opt.strats.values()
            if hasattr(s, "strategies")
        ),
        "timings": opt.profile.get("timings", {}),
        "ilp": opt.profile.get("ilp", {}),
    }


def _eval_shape(task: dict[str, Any]) -> dict[str, Any]:
    shape = tuple(task["shape"])
    seedball_loaded = task.get("seedball")
    result: dict[str, Any] = {
        "shape": list(shape),
        "mesh_dim_names": list(_mesh_dim_names(len(shape))),
    }
    try:
        trace_t0 = time.perf_counter()
        _autop, gm = _trace_candidate(shape)
        result["trace_s"] = time.perf_counter() - trace_t0
    except Exception as exc:  # noqa: BLE001
        result["feasible"] = False
        result["error"] = f"trace {type(exc).__name__}: {exc}"
        return result

    try:
        if seedball_loaded is None or _G["recompute_seedball"]:
            seedball = _compute_seedball(gm, shape, _G["max_time_s"])
        else:
            seedball = dict(seedball_loaded)
        result["seedball_trws_lazy"] = seedball
    except Exception as exc:  # noqa: BLE001
        result["seedball_trws_lazy"] = {
            "feasible": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        result["full_lp_relaxation"] = _solve_full_lp(gm, shape)
    except Exception as exc:  # noqa: BLE001
        result["full_lp_relaxation"] = {
            "feasible": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    seedball_obj = result.get("seedball_trws_lazy", {}).get("objective")
    lp_obj = result.get("full_lp_relaxation", {}).get("objective")
    if seedball_obj is not None and lp_obj is not None and math.isfinite(lp_obj):
        abs_gap = float(seedball_obj) - float(lp_obj)
        rel_gap = abs_gap / max(abs(float(lp_obj)), 1.0)
        result["gap"] = {
            "absolute": abs_gap,
            "relative": rel_gap,
            "trws_below_lp": abs_gap < -_G["tolerance"],
        }
    return result


def _write_report(payload: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# DSV3 Seed-Ball TRW-S vs Full LP Relaxation",
        "",
        f"JSON: `{payload['output_json']}`",
        "",
        "| mesh | TRW-S lazy | full LP | abs gap | rel gap | LP status | LP vars | constraints |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in payload["results"]:
        shape = tuple(row["shape"])
        trws = row.get("seedball_trws_lazy", {})
        lp = row.get("full_lp_relaxation", {})
        gap = row.get("gap", {})
        lines.append(
            f"| `{shape}` | "
            f"{_fmt_obj(trws.get('objective'))} | "
            f"{_fmt_obj(lp.get('objective'))} | "
            f"{_fmt_obj(gap.get('absolute'))} | "
            f"{_fmt_pct(gap.get('relative'))} | "
            f"{lp.get('status', lp.get('error', 'NA'))} | "
            f"{lp.get('n_vars', 'NA')} | {lp.get('constraints', 'NA')} |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_obj(value: Any) -> str:
    return f"{value:.1f}" if isinstance(value, (int, float)) else "NA"


def _fmt_pct(value: Any) -> str:
    return f"{value:.4%}" if isinstance(value, (int, float)) else "NA"


def _print_results(results: list[dict[str, Any]]) -> None:
    print(
        f"{'mesh':>18} {'trws':>14} {'lp':>14} {'abs_gap':>14} "
        f"{'rel_gap':>10} {'lp_status':>10} {'lp_vars':>10}",
        flush=True,
    )
    for row in results:
        shape = tuple(row["shape"])
        trws = row.get("seedball_trws_lazy", {})
        lp = row.get("full_lp_relaxation", {})
        gap = row.get("gap", {})
        print(
            f"{str(shape):>18} {_fmt_obj(trws.get('objective')):>14} "
            f"{_fmt_obj(lp.get('objective')):>14} "
            f"{_fmt_obj(gap.get('absolute')):>14} "
            f"{_fmt_pct(gap.get('relative')):>10} "
            f"{lp.get('status', 'ERR'):>10} {str(lp.get('n_vars', 'NA')):>10}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=128)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--local-batch-size", type=int, default=8)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-time-s", type=float, default=600.0)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--allow-4d", action="store_true")
    parser.add_argument("--recompute-seedball", action="store_true")
    parser.add_argument("--verbose-lp", action="store_true")
    parser.add_argument(
        "--seedball-json",
        default="out/meshdisc/dsv3_seed_vs_final_ws128.json",
        help="Existing seed-ball result JSON to use for TRW-S objectives.",
    )
    parser.add_argument(
        "--only-shape",
        type=parse_shape,
        action="append",
        help="Restrict to comma-separated shapes.",
    )
    parser.add_argument(
        "--output-json",
        default="out/meshdisc/dsv3_lp_compare_ws128.json",
    )
    parser.add_argument("--output-report", default=None)
    args = parser.parse_args()

    shapes = args.only_shape or list(DEFAULT_SHAPES)
    shapes = [tuple(shape) for shape in shapes]
    if not args.allow_4d:
        four_d = [shape for shape in shapes if len(shape) > 3]
        if four_d:
            raise ValueError(f"4D shapes are disabled by default: {four_d}")
    bad_product = [shape for shape in shapes if math.prod(shape) != args.world_size]
    if bad_product:
        raise ValueError(f"shape product must match world size: {bad_product}")

    seedball_by_shape = _load_seedball(args.seedball_json)
    tasks = [
        {"shape": shape, "seedball": seedball_by_shape.get(shape)}
        for shape in shapes
    ]
    cfg = {
        "world": args.world_size,
        "gpn": args.gpus_per_node,
        "seq": args.seq_len,
        "radius": args.radius,
        "max_time_s": args.max_time_s,
        "local_batch_size": args.local_batch_size,
        "num_experts": args.num_experts,
        "recompute_seedball": args.recompute_seedball,
        "verbose_lp": args.verbose_lp,
        "tolerance": args.tolerance,
    }

    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    results = []
    with ctx.Pool(
        processes=min(args.workers, len(tasks)),
        initializer=_init_worker,
        initargs=(cfg,),
        maxtasksperchild=1,
    ) as pool:
        for row in pool.imap_unordered(_eval_shape, tasks, chunksize=1):
            results.append(row)
            trws = row.get("seedball_trws_lazy", {})
            lp = row.get("full_lp_relaxation", {})
            print(
                f"done shape={tuple(row['shape'])} "
                f"trws={trws.get('objective')} lp={lp.get('objective')} "
                f"lp_status={lp.get('status', lp.get('error'))}",
                flush=True,
            )

    order = {shape: idx for idx, shape in enumerate(shapes)}
    results.sort(key=lambda row: order[tuple(row["shape"])])
    payload = {
        "config": {
            "model": "DeepSeekV3Model from examples/example_ds3_local_map.py",
            "world_size": args.world_size,
            "gpus_per_node": args.gpus_per_node,
            "seq": args.seq_len,
            "num_experts": args.num_experts,
            "local_batch_size": args.local_batch_size,
            "radius": args.radius,
            "seedball_json": args.seedball_json,
            "recompute_seedball": args.recompute_seedball,
            "shapes": [list(shape) for shape in shapes],
        },
        "wall_s": time.perf_counter() - t0,
        "results": results,
    }
    payload["output_json"] = args.output_json
    write_json(payload, args.output_json)

    report = args.output_report
    if report is None:
        report = str(args.output_json).removesuffix(".json") + ".md"
    _write_report(payload, report)

    _print_results(results)
    print(f"saved {args.output_json}", flush=True)
    print(f"saved {report}", flush=True)


if __name__ == "__main__":
    main()
