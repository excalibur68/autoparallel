# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import math
from pathlib import Path

import torch
from torch.testing._comparison import default_tolerances

CASES = (
    "plain-sharded",
    "plain-replicated-counts",
    "flex-default",
    "flex-replicated-counts",
)


def _load_case(root, case, rank):
    rank_dir = root / case / "0" / f"rank_{rank}"
    return {
        "model_state": torch.load(
            rank_dir / "model_state.pt", map_location="cpu", weights_only=True
        ),
        "global_model_state": torch.load(
            rank_dir / "global_model_state.pt",
            map_location="cpu",
            weights_only=True,
        ),
        "full_batch": torch.load(
            rank_dir / "full_batch.pt", map_location="cpu", weights_only=True
        ),
        "outputs": [
            torch.load(
                rank_dir / f"mb{index}_output.pt",
                map_location="cpu",
                weights_only=True,
            )
            for index in range(16)
        ],
        "gradients": torch.load(
            rank_dir / "gradients.pt", map_location="cpu", weights_only=True
        ),
        "global_gradients": torch.load(
            rank_dir / "global_gradients.pt", map_location="cpu", weights_only=True
        ),
        "final_buffers": torch.load(
            rank_dir / "final_buffers.pt", map_location="cpu", weights_only=True
        ),
        "global_final_buffers": torch.load(
            rank_dir / "global_final_buffers.pt",
            map_location="cpu",
            weights_only=True,
        ),
    }


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _flatten(value[key], f"{prefix}/{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _flatten(item, f"{prefix}/{index}")
    else:
        yield prefix, value


def _compare_tensors(actual, expected, exact):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False, math.inf, math.inf, "shape or dtype mismatch"
    if torch.equal(actual, expected):
        return True, 0.0, 0.0, None

    rtol, atol = (0.0, 0.0) if exact else default_tolerances(actual, expected)
    actual = actual.flatten()
    expected = expected.flatten()
    chunk_size = 4 * 1024 * 1024
    mismatches = 0
    max_abs = 0.0
    max_rel = 0.0
    for start in range(0, actual.numel(), chunk_size):
        lhs = actual[start : start + chunk_size]
        rhs = expected[start : start + chunk_size]
        difference = (lhs - rhs).abs().float()
        mismatches += (~torch.isclose(lhs, rhs, rtol=rtol, atol=atol)).sum().item()
        if difference.numel():
            max_abs = max(max_abs, difference.max().item())
            denominator = rhs.abs().float()
            relative = torch.where(
                denominator == 0,
                torch.where(difference == 0, 0.0, torch.inf),
                difference / denominator,
            )
            max_rel = max(max_rel, relative.max().item())
    error = None
    if mismatches:
        error = f"{mismatches} values outside rtol={rtol}, atol={atol}"
    return not mismatches, max_abs, max_rel, error


def _compare_component(actual, expected, exact):
    actual_entries = dict(_flatten(actual))
    expected_entries = dict(_flatten(expected))
    names = sorted(set(actual_entries) | set(expected_entries))
    failures = []
    max_abs = 0.0
    max_rel = 0.0
    for name in names:
        if name not in actual_entries or name not in expected_entries:
            failures.append({"name": name, "error": "missing entry"})
            continue
        lhs = actual_entries[name]
        rhs = expected_entries[name]
        passed, entry_abs, entry_rel, error = _compare_tensors(lhs, rhs, exact)
        if not passed:
            failures.append({"name": name, "error": error})
        max_abs = max(max_abs, entry_abs)
        max_rel = max(max_rel, entry_rel)
    return {
        "entries": len(names),
        "failures": failures,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "passed": not failures,
    }


def _compare_pair(
    root, actual_case, expected_case, exact, world_size, component_names=None
):
    ranks = []
    for rank in range(world_size):
        actual = _load_case(root, actual_case, rank)
        expected = _load_case(root, expected_case, rank)
        names = component_names or tuple(actual)
        components = {
            name: _compare_component(actual[name], expected[name], exact)
            for name in names
        }
        ranks.append(
            {
                "rank": rank,
                "components": components,
                "passed": all(result["passed"] for result in components.values()),
            }
        )
    return {
        "actual": actual_case,
        "expected": expected_case,
        "comparison": "exact" if exact else "torch.testing.assert_close defaults",
        "ranks": ranks,
        "passed": all(rank["passed"] for rank in ranks),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()

    missing = [case for case in CASES if not (args.results_dir / case / "0").is_dir()]
    if missing:
        raise FileNotFoundError(f"missing case results: {missing}")

    comparisons = [
        _compare_pair(
            args.results_dir,
            "flex-default",
            "plain-sharded",
            exact=True,
            world_size=args.world_size,
        ),
        _compare_pair(
            args.results_dir,
            "flex-replicated-counts",
            "plain-replicated-counts",
            exact=True,
            world_size=args.world_size,
        ),
        _compare_pair(
            args.results_dir,
            "plain-replicated-counts",
            "plain-sharded",
            exact=False,
            world_size=args.world_size,
            component_names=(
                "global_model_state",
                "full_batch",
                "outputs",
                "global_gradients",
                "global_final_buffers",
            ),
        ),
    ]
    result = {
        "comparisons": comparisons,
        "all_passed": all(comparison["passed"] for comparison in comparisons),
    }
    output_path = args.results_dir / "numerics_comparison.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
