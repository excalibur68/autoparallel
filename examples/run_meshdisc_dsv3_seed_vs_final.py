#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Compare raw seed vs seed-ball TRW-S ranking for DeepSeekV3 MoE meshes."""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import queue
import time
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from meshdisc_rank_utils import (
    attach_ranks,
    n_strats,
    parse_shape,
    print_summary,
    print_table,
    solution_json,
    write_json,
    write_rank_report,
    write_summary_csv,
)
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Partial, Placement, Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel._testing.models.dsv3 import DeepSeekV3Model, make_dsv3_config
from autoparallel.api import AutoParallel
from autoparallel.approximate_sharding import ApproximateShardingSolver
from autoparallel.cost_models.collective_runtime_estimation import (
    get_nccl_topo_config,
    set_nccl_topo_config,
)
from autoparallel.cost_models.nccl_cost_model import h100_topo_config
from autoparallel.mesh_search import (
    _factored_seed_cache_key,
    _factored_seed_dim_cost_model,
    _set_cost_model_for_mesh,
    fabric_exponents_from_nccl_topo,
    generate_fabric_mesh_shapes,
    reset_mesh_search_caches,
    seed_ball_approx_kwargs,
)
from autoparallel.optimize_sharding import ShardingOptimizer


_G: dict[str, Any] = {}


def _mesh_dim_names(ndim: int) -> tuple[str, ...]:
    if ndim < 2:
        raise ValueError("DeepSeekV3 MoE local_map runner requires at least 2D mesh")
    names = ("dp", "ep", "d2", "d3")
    return names[:ndim]


def _is_placement_tuple(value: Any) -> bool:
    return isinstance(value, (tuple, list)) and all(
        isinstance(p, Placement) for p in value
    )


def _degenerate_placement(placement: Placement, mesh_size: int) -> Placement:
    if mesh_size == 1 and isinstance(placement, (Partial, Shard)):
        return Replicate()
    return placement


def _expand_placement_spec(spec: Any, mesh_shape: tuple[int, ...]) -> Any:
    if spec is None:
        return None
    ndim = len(mesh_shape)
    if isinstance(spec, Placement):
        placements = (spec,)
    elif _is_placement_tuple(spec):
        placements = tuple(spec)
    else:
        raise TypeError(f"unsupported local_map placement spec: {spec!r}")
    if len(placements) > ndim:
        raise ValueError(
            f"local_map placement rank {len(placements)} exceeds mesh ndim {ndim}"
        )
    expanded = placements + (Replicate(),) * (ndim - len(placements))
    return tuple(
        _degenerate_placement(placement, int(mesh_shape[idx]))
        for idx, placement in enumerate(expanded)
    )


def _project_placement_spec(
    spec: Any,
    mesh_shape: tuple[int, ...],
    dim_idx: int,
) -> Any:
    expanded = _expand_placement_spec(spec, mesh_shape)
    if expanded is None:
        return None
    return (expanded[dim_idx],)


def _map_local_map_placements(
    value: Any,
    mesh_shape: tuple[int, ...],
    dim_idx: int | None,
) -> Any:
    if value is None:
        return None
    mapped = []
    for spec in value:
        if dim_idx is None:
            mapped.append(_expand_placement_spec(spec, mesh_shape))
        else:
            mapped.append(_project_placement_spec(spec, mesh_shape, dim_idx))
    return tuple(mapped)


def _normalize_local_map_metadata(
    gm: torch.fx.GraphModule,
    mesh_shape: tuple[int, ...],
    *,
    dim_idx: int | None = None,
) -> None:
    for node in gm.graph.nodes:
        kwargs = node.meta.get("local_map_kwargs")
        if not kwargs:
            continue
        kwargs["device_mesh"] = None
        kwargs["in_placements"] = _map_local_map_placements(
            kwargs["in_placements"], mesh_shape, dim_idx
        )
        kwargs["out_placements"] = _map_local_map_placements(
            kwargs["out_placements"], mesh_shape, dim_idx
        )
        if kwargs.get("in_grad_placements") is not None:
            kwargs["in_grad_placements"] = _map_local_map_placements(
                kwargs["in_grad_placements"], mesh_shape, dim_idx
            )


def _snapshot_local_map_metadata(gm: torch.fx.GraphModule) -> list[tuple[Any, dict]]:
    snapshot = []
    for node in gm.graph.nodes:
        kwargs = node.meta.get("local_map_kwargs")
        if kwargs:
            snapshot.append((node, dict(kwargs)))
    return snapshot


def _restore_local_map_metadata(snapshot: list[tuple[Any, dict]]) -> None:
    for node, kwargs in snapshot:
        node.meta["local_map_kwargs"] = dict(kwargs)


@contextmanager
def _local_map_metadata(
    gm: torch.fx.GraphModule,
    mesh_shape: tuple[int, ...],
    *,
    dim_idx: int | None,
) -> Iterator[None]:
    snapshot = _snapshot_local_map_metadata(gm)
    try:
        _normalize_local_map_metadata(gm, mesh_shape, dim_idx=dim_idx)
        yield
    finally:
        _restore_local_map_metadata(snapshot)


def _first_output_placements(output_specs: Any) -> tuple[Placement, ...] | None:
    from torch.distributed.tensor._dtensor_spec import DTensorSpec

    if isinstance(output_specs, DTensorSpec):
        return tuple(output_specs.placements)
    if isinstance(output_specs, (list, tuple)):
        for spec in output_specs:
            placements = _first_output_placements(spec)
            if placements is not None:
                return placements
    return None


def _repair_duplicate_tensor_dim_shards(
    seed: dict[str, tuple[Placement, ...]],
    protected_nodes: set[str],
) -> int:
    """Keep raw DSV3 seeds feasible after independently solving mesh dims.

    With num_experts=64, independent 1D solves can shard the same tensor
    dimension on multiple mesh dimensions.  Some DTensor view rules cannot
    propagate those stacked placements through the full graph.  Keep the
    innermost/rightmost shard and relax the outer duplicates to replicate;
    explicit IO constraints are protected and remain exactly as requested.
    """

    changed = 0
    for node_name, placements in list(seed.items()):
        if node_name in protected_nodes:
            continue
        repaired = list(placements)
        seen_dims: set[int] = set()
        for idx in range(len(repaired) - 1, -1, -1):
            placement = repaired[idx]
            if not isinstance(placement, Shard):
                continue
            if placement.dim in seen_dims:
                repaired[idx] = Replicate()
            else:
                seen_dims.add(placement.dim)
        repaired_tuple = tuple(repaired)
        if repaired_tuple != placements:
            seed[node_name] = repaired_tuple
            changed += 1
    return changed


def _build_dsv3_factored_seed(
    gm: torch.fx.GraphModule,
    mesh_shape: tuple[int, ...],
    input_placements: tuple[Placement, ...],
) -> tuple[dict[str, tuple[Placement, ...]], int]:
    from torch._functorch._aot_autograd.fx_utils import (
        get_plain_input_and_grad_nodes,
        get_plain_output_and_tangent_nodes,
    )

    ndim = len(mesh_shape)
    if len(input_placements) != ndim:
        raise ValueError(
            f"input_placements has {len(input_placements)} entries, expected {ndim}"
        )
    per_dim: list[dict[str, Placement]] = []
    for dim_idx, size in enumerate(mesh_shape):
        input_pl = input_placements[dim_idx]
        if int(size) == 1:
            per_dim.append({})
            continue
        key = _factored_seed_cache_key(
            int(size),
            input_pl,
            _G["topo"],
            mesh_shape,
            dim_idx,
            fabric_aware=True,
        )
        if key not in _G["seed_cache"]:
            with unset_fake_temporarily():
                mesh_1d = torch.distributed.device_mesh.init_device_mesh(
                    "cuda", (int(size),), mesh_dim_names=("d",)
                )
            prev = get_nccl_topo_config()
            try:
                dim_cost_model = _factored_seed_dim_cost_model(
                    _G["topo"], mesh_shape, dim_idx, fabric_aware=True
                )
                _set_cost_model_for_mesh(mesh_1d, dim_cost_model)
                reset_mesh_search_caches()
                with _local_map_metadata(gm, mesh_shape, dim_idx=dim_idx):
                    opt = ShardingOptimizer(
                        gm,
                        mesh_1d,
                        _G["force_grad_reduce"],
                        repeated_subgraphs=True,
                    )
                    opt.add_sharded_input_constraint([(input_pl,)])
                    opt.add_sharded_output_constraint([(input_pl,)])
                    opt.add_parameter_memory_constraint(0.0, 1.0 / int(size))
                    solution = opt.get_solution()
            finally:
                set_nccl_topo_config(prev)

            node_pl: dict[str, Placement] = {}
            for node, strategy in solution.items():
                placements = _first_output_placements(strategy.output_specs)
                if placements is not None:
                    node_pl[node.name] = placements[0]
            _G["seed_cache"][key] = node_pl
        per_dim.append(_G["seed_cache"][key])

    seed: dict[str, tuple[Placement, ...]] = {}
    for node in gm.graph.nodes:
        if node.op == "output":
            continue
        seed[node.name] = tuple(
            per_dim[dim_idx].get(node.name, Replicate()) for dim_idx in range(ndim)
        )

    io_pl = tuple(input_placements)
    protected_nodes: set[str] = set()
    for getter in (get_plain_input_and_grad_nodes, get_plain_output_and_tangent_nodes):
        for _desc, (node, comp) in getter(gm.graph).items():
            seed[node.name] = io_pl
            protected_nodes.add(node.name)
            if comp is not None:
                seed[comp.name] = io_pl
                protected_nodes.add(comp.name)
    repair_count = _repair_duplicate_tensor_dim_shards(seed, protected_nodes)
    return seed, repair_count


def _input_sharding(shape: tuple[int, ...]) -> tuple[Placement, ...]:
    ndim = len(shape)
    if ndim < 2:
        raise ValueError("DeepSeekV3 MoE input sharding requires dp and ep axes")
    placements: list[Placement] = [Shard(0), Shard(0)] + [Replicate()] * (ndim - 2)
    return tuple(
        _degenerate_placement(placement, int(shape[idx]))
        for idx, placement in enumerate(placements)
    )


def _trace_candidate(shape: tuple[int, ...]) -> tuple[Any, torch.fx.GraphModule]:
    ndim = len(shape)
    names = _mesh_dim_names(ndim)
    with unset_fake_temporarily():
        mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda", shape, mesh_dim_names=names
        )
    config = make_dsv3_config(
        num_experts=_G["num_experts"],
        max_seq_len=_G["seq"],
    )
    with torch.device("meta"):
        model = DeepSeekV3Model(
            config,
            mesh=mesh,
            compute_dtype=torch.bfloat16,
        )
    for module in model.modules():
        if hasattr(module, "axis_name"):
            module.axis_name = "ep"

    global_batch_size = _G["local_batch_size"] * _G["world"]

    def input_fn():
        return torch.randint(
            0,
            config.vocab_size,
            (global_batch_size, config.rope.max_seq_len),
            device="cuda",
        )

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    autop = AutoParallel(
        model,
        input_fn,
        mesh,
        mp_policy=mp_policy,
        dynamic=True,
        repeated_subgraphs=True,
    )
    prev = get_nccl_topo_config()
    try:
        set_nccl_topo_config(_G["topo"])
        with mesh:
            autop.build_model_graph()
        _normalize_local_map_metadata(autop.gm, shape)
    finally:
        set_nccl_topo_config(prev)
    return autop, autop.gm


def _build_optimizer(
    gm: torch.fx.GraphModule,
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
        gm,
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
    gm: torch.fx.GraphModule,
    shape: tuple[int, ...],
    seed: dict[str, tuple[Any, ...]],
    radius: int,
    x_sharding: tuple[Any, ...],
    max_time_s: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    opt = _build_optimizer(gm, shape, seed, radius, x_sharding)
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
        "strategy": solution_json(solution, gm),
    }
    if not payload["feasible"]:
        payload["error"] = "non-finite objective"
    return payload


def _worker_init(cfg: dict[str, Any]) -> None:
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


def _eval_candidate(shape: tuple[int, ...]) -> dict[str, Any]:
    ndim = len(shape)
    x_sharding = _input_sharding(shape)
    result: dict[str, Any] = {
        "shape": list(shape),
        "mesh_dim_names": list(_mesh_dim_names(ndim)),
    }
    autop = None
    try:
        trace_t0 = time.perf_counter()
        autop, gm = _trace_candidate(shape)
        result["trace_s"] = time.perf_counter() - trace_t0
    except Exception as exc:  # noqa: BLE001
        result["trace_error"] = f"{type(exc).__name__}: {exc}"
        if _G.get("raw_queue") is not None:
            _G["raw_queue"].put(
                {
                    "shape": list(shape),
                    "feasible": False,
                    "error": result["trace_error"],
                    "phase": "trace",
                }
            )
        return result

    try:
        t0 = time.perf_counter()
        seed, repair_count = _build_dsv3_factored_seed(gm, shape, x_sharding)
        seed_s = time.perf_counter() - t0
        result["seed_s"] = seed_s
        result["seed_repair_changes"] = repair_count
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
            gm,
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
            gm,
            shape,
            seed,
            _G["radius"],
            x_sharding,
            max_time_s=_G["max_time_s"],
        )
        final_score["seed_s"] = seed_s
        final_score["trace_s"] = result["trace_s"]
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
        "radius": args.radius,
        "max_time_s": args.max_time_s,
        "local_batch_size": args.local_batch_size,
        "num_experts": args.num_experts,
    }
    workers = min(args.workers, len(shapes))
    fabrics = _fabric_exponents(world_size, args.gpus_per_node)
    print(
        f"\nWS{world_size} DSV3 seed-vs-final: candidates={len(shapes)} "
        f"workers={workers} fabrics={fabrics} "
        f"seq={args.seq_len} num_experts={args.num_experts} "
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
    parser.add_argument("--world-size", type=int, default=128)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--local-batch-size", type=int, default=8)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--max-time-s", type=float, default=600.0)
    parser.add_argument(
        "--only-shape",
        type=parse_shape,
        action="append",
        help="Restrict to comma-separated canonical shapes.",
    )
    parser.add_argument(
        "--output-json",
        default="out/meshdisc/dsv3_seed_vs_final_ws128.json",
    )
    parser.add_argument("--output-report", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    if args.gpus_per_node != 8:
        raise ValueError("This runner is scoped to nodesize=8 NVLink + RDMA.")
    if args.world_size < args.gpus_per_node or args.world_size & (args.world_size - 1):
        raise ValueError("world_size must be a power of two and at least 8")

    topo = h100_topo_config(
        num_nodes=max(1, args.world_size // args.gpus_per_node),
        gpus_per_node=args.gpus_per_node,
    )
    fabrics = fabric_exponents_from_nccl_topo(args.world_size, topo)
    shapes = generate_fabric_mesh_shapes(args.world_size, fabrics)
    shapes = [shape for shape in shapes if len(shape) >= 2]
    if args.only_shape:
        selected = [s for s in args.only_shape if math.prod(s) == args.world_size]
        allowed = set(shapes)
        unknown = [shape for shape in selected if shape not in allowed]
        if unknown:
            raise ValueError(
                f"--only-shape contains non-canonical DSV3 shapes: {unknown}"
            )
        shapes = list(dict.fromkeys(selected))
    if not shapes:
        raise ValueError("no candidate shapes selected")

    ctx = mp.get_context("spawn")
    all_t0 = time.perf_counter()
    world_results = [_run_world_size(args, args.world_size, shapes, ctx)]
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
            "model": "DeepSeekV3Model from examples/example_ds3_local_map.py",
            "world_size": args.world_size,
            "gpus_per_node": args.gpus_per_node,
            "seq": args.seq_len,
            "num_experts": args.num_experts,
            "local_batch_size": args.local_batch_size,
            "global_batch_size": args.local_batch_size * args.world_size,
            "radius": args.radius,
            "max_time_s": args.max_time_s,
            "workers": args.workers,
            "input_output_sharding": "(Shard(0), Shard(0)) + Replicate()",
            "mesh_dim_names": "('dp', 'ep', 'd2', 'd3')[:ndim]",
            "topology": "h100_topo_config, NVLink8 + RDMA",
            "dsv3_seed_repair": (
                "For non-IO nodes, duplicate shards of the same tensor dimension "
                "from independent 1D seed solves are relaxed to Replicate on the "
                "outer mesh dims, keeping the rightmost shard. Size-1 mesh dims "
                "also degenerate Shard/Partial to Replicate."
            ),
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
        title="MeshDisc DeepSeekV3 MoE Seed vs Final Rank Report",
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
