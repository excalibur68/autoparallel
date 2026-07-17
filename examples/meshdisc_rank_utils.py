# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.placement_types import Replicate, Shard


def parse_world_sizes(text: str) -> list[int]:
    try:
        sizes = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid world size list {text!r}") from exc
    if not sizes:
        raise argparse.ArgumentTypeError("world size list must be non-empty")
    bad = [ws for ws in sizes if ws < 1 or ws & (ws - 1)]
    if bad:
        raise argparse.ArgumentTypeError(f"world sizes must be powers of two: {bad}")
    return sizes


def parse_shape(text: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid shape {text!r}") from exc
    if not shape:
        raise argparse.ArgumentTypeError(f"shape must be non-empty: {text!r}")
    return shape


def placement_str(placement: Any) -> str:
    if isinstance(placement, Shard):
        return f"S({placement.dim})"
    if isinstance(placement, Replicate):
        return "R"
    return str(placement)


def spec_json(spec: Any) -> Any:
    if isinstance(spec, DTensorSpec):
        return [placement_str(p) for p in spec.placements]
    if isinstance(spec, (tuple, list)):
        return [spec_json(x) for x in spec]
    if spec is None:
        return None
    return str(spec)


def solution_json(solution: dict[Any, Any] | None, gm: Any) -> list[dict[str, Any]]:
    if solution is None:
        return []
    rows: list[dict[str, Any]] = []
    graph_nodes = list(gm.graph.nodes)
    node_order = {node: i for i, node in enumerate(graph_nodes)}
    for node, strategy in sorted(
        solution.items(), key=lambda kv: node_order.get(kv[0], len(node_order))
    ):
        rows.append(
            {
                "node": node.name,
                "op": node.op,
                "target": str(node.target),
                "output_placements": spec_json(
                    getattr(strategy, "output_specs", None)
                ),
                "input_placements": spec_json(getattr(strategy, "input_specs", None)),
            }
        )
    return rows


def n_strats(opt: Any) -> int:
    return sum(
        len(s.strategies) for s in opt.strats.values() if hasattr(s, "strategies")
    )


def average_ranks(
    items: list[dict[str, Any]],
    key: str,
) -> dict[tuple[int, ...], float]:
    vals = []
    for item in items:
        value = item.get(key, {}).get("objective")
        if value is not None and math.isfinite(value):
            vals.append((tuple(item["shape"]), float(value)))
    vals.sort(key=lambda x: x[1])

    ranks: dict[tuple[int, ...], float] = {}
    i = 0
    while i < len(vals):
        j = i + 1
        while j < len(vals) and vals[j][1] == vals[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[vals[k][0]] = avg_rank
        i = j
    return ranks


def spearman(
    raw_ranks: dict[tuple[int, ...], float],
    final_ranks: dict[tuple[int, ...], float],
) -> dict[str, float | int]:
    shapes = sorted(set(raw_ranks) & set(final_ranks))
    n = len(shapes)
    if n < 2:
        return {"rho": float("nan"), "n": n}
    xs = [raw_ranks[s] for s in shapes]
    ys = [final_ranks[s] for s in shapes]
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    rho = cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")
    return {"rho": rho, "n": n}


def attach_ranks(results: list[dict[str, Any]]) -> dict[str, Any]:
    raw_ranks = average_ranks(results, "raw_seed")
    final_ranks = average_ranks(results, "final")
    by_shape = {tuple(r["shape"]): r for r in results}

    for shape, rank in raw_ranks.items():
        by_shape[shape]["raw_seed"]["rank"] = rank
    for shape, rank in final_ranks.items():
        by_shape[shape]["final"]["rank"] = rank

    final_sorted = sorted(
        (r for r in results if tuple(r["shape"]) in final_ranks),
        key=lambda r: final_ranks[tuple(r["shape"])],
    )
    best = None
    if final_sorted:
        row = final_sorted[0]
        shape = tuple(row["shape"])
        best = {
            "shape": list(shape),
            "final_rank": final_ranks[shape],
            "raw_seed_rank": raw_ranks.get(shape),
            "final_objective": row["final"]["objective"],
            "raw_seed_objective": row.get("raw_seed", {}).get("objective"),
        }
    return {
        "raw_ranks": raw_ranks,
        "final_ranks": final_ranks,
        "spearman": spearman(raw_ranks, final_ranks),
        "best": best,
    }


def print_table(title: str, results: list[dict[str, Any]], key: str) -> None:
    ranked = [r for r in results if r.get(key, {}).get("feasible")]
    ranked.sort(key=lambda r: r[key]["rank"])
    print(f"\n=== {title} ===", flush=True)
    print(
        f"{'rank':>5} {'mesh':>18} {'objective':>16} "
        f"{'build_s':>8} {'solve_s':>8} {'n_strats':>9}",
        flush=True,
    )
    for row in ranked:
        item = row[key]
        print(
            f"{item['rank']:5.1f} {str(tuple(row['shape'])):>18} "
            f"{item['objective']:16.1f} {item['build_s']:8.1f} "
            f"{item['solve_s']:8.1f} {item['n_strats']:9d}",
            flush=True,
        )
    for row in results:
        item = row.get(key, {})
        if not item.get("feasible", True):
            print(
                f"{str(tuple(row['shape'])):>18} INFEASIBLE {item.get('error')}",
                flush=True,
            )
        elif key not in row:
            error = row.get("seed_error") or row.get("trace_error") or "missing result"
            print(
                f"{str(tuple(row['shape'])):>18} INFEASIBLE {error}",
                flush=True,
            )


def print_summary(summary: list[dict[str, Any]]) -> None:
    print("\n=== summary ===", flush=True)
    print(
        f"{'ws':>6} {'n':>4} {'raw_rank_s':>12} {'wall_s':>10} "
        f"{'rho':>10} {'best_final':>18} {'best_raw_rank':>13}",
        flush=True,
    )
    for row in summary:
        best = row["best_final"]
        best_shape = tuple(best["shape"]) if best is not None else None
        best_raw_rank = best.get("raw_seed_rank") if best is not None else None
        best_raw_rank_s = (
            f"{best_raw_rank:13.1f}" if best_raw_rank is not None else f"{'NA':>13}"
        )
        print(
            f"{row['world_size']:6d} {row['n_shapes']:4d} "
            f"{row['raw_rank_ready_s']:12.1f} {row['wall_s']:10.1f} "
            f"{row['spearman']['rho']:10.6f} {str(best_shape):>18} "
            f"{best_raw_rank_s}",
            flush=True,
        )


def ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    out = ensure_parent(path)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary_csv(payload: dict[str, Any], path: str | Path) -> None:
    out = ensure_parent(path)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "world_size",
                "n_shapes",
                "raw_rank_ready_s",
                "wall_s",
                "spearman",
                "best_final_shape",
                "best_final_objective",
                "best_raw_seed_rank",
                "best_raw_seed_objective",
            ]
        )
        for row in payload["summary"]:
            best = row["best_final"] or {}
            writer.writerow(
                [
                    row["world_size"],
                    row["n_shapes"],
                    row["raw_rank_ready_s"],
                    row["wall_s"],
                    row["spearman"]["rho"],
                    tuple(best.get("shape", [])),
                    best.get("final_objective"),
                    best.get("raw_seed_rank"),
                    best.get("raw_seed_objective"),
                ]
            )


def write_rank_report(
    payload: dict[str, Any],
    path: str | Path,
    *,
    title: str,
    full_json_path: str | Path | None = None,
) -> None:
    out = ensure_parent(path)
    lines = [f"# {title}", ""]
    if full_json_path is not None:
        lines += [f"Full JSON: `{full_json_path}`", ""]

    lines += [
        "## Summary",
        "",
        (
            "| world_size | n_shapes | raw_rank_ready_s | wall_s | "
            "spearman | TRW-S rank1 | raw rank |"
        ),
        "|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in payload["summary"]:
        best = row["best_final"] or {}
        best_shape = tuple(best.get("shape", []))
        raw_rank = best.get("raw_seed_rank")
        raw_rank_s = f"{raw_rank:.1f}" if raw_rank is not None else "NA"
        lines.append(
            f"| {row['world_size']} | {row['n_shapes']} | "
            f"{row['raw_rank_ready_s']:.1f} | {row['wall_s']:.1f} | "
            f"{row['spearman']['rho']:.6f} | `{best_shape}` | {raw_rank_s} |"
        )

    for world in payload["world_results"]:
        best = world["best_final"] or {}
        best_shape = tuple(best.get("shape", []))
        raw_rank = best.get("raw_seed_rank")
        raw_rank_s = f"{raw_rank:.1f}" if raw_rank is not None else "NA"
        lines += [
            "",
            f"## WS{world['world_size']}",
            "",
            (
                f"Spearman: `{world['spearman']['rho']:.6f}`; "
                f"raw rank ready: `{world['raw_rank_ready_s']:.1f}s`; "
                f"best final: `{best_shape}`; "
                f"best final raw rank: `{raw_rank_s}`"
            ),
            "",
            (
                "| raw rank | raw objective | final rank | final objective | "
                "mesh |"
            ),
            "|---:|---:|---:|---:|---|",
        ]
        rows = [
            r
            for r in world["results"]
            if r.get("raw_seed", {}).get("feasible")
            and r.get("final", {}).get("feasible")
        ]
        rows.sort(key=lambda r: r["final"]["rank"])
        for row in rows:
            lines.append(
                f"| {row['raw_seed']['rank']:.1f} | "
                f"{row['raw_seed']['objective']:.1f} | "
                f"{row['final']['rank']:.1f} | "
                f"{row['final']['objective']:.1f} | "
                f"`{tuple(row['shape'])}` |"
            )

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
