# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace
from itertools import combinations, product
from typing import Any, Callable, Optional

import pulp
import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard
from torch.utils._pytree import tree_flatten

from .cost_models.collective_runtime_estimation import (
    get_nccl_topo_config,
    reset_comms_cost_cache,
    set_nccl_topo_config,
)
from .cost_models.compute_estimation import reset_compute_cost_cache
from .cost_models.nccl_cost_model import (
    NCCLTopoConfig,
    derive_mesh_dim_topo,
    detect_nccl_topo_config,
)
from .optimize_sharding import ShardingOptimizer
from .shardings.placement_options import reset_placement_options_cache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MeshCandidate:
    """A semantic mesh candidate.

    Non-DP semantic axes such as TP, CP, and EP are represented as rightmost
    dimensions so heavy communication maps to the fastest topology tier.
    """

    mesh_shape: tuple[int, ...]
    mesh_dim_names: tuple[str, ...]
    dp_size: int
    tp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1
    score: float = 0.0
    reason: str = ""

    @property
    def ndim(self) -> int:
        return len(self.mesh_shape)

    @property
    def world_size(self) -> int:
        return math.prod(self.mesh_shape)

    @property
    def semantic_axis_sizes(self) -> dict[str, int]:
        all_sizes = {
            "dp": self.dp_size,
            "tp": self.tp_size,
            "cp": self.cp_size,
            "ep": self.ep_size,
        }
        return {
            name: all_sizes[name]
            for name in self.mesh_dim_names
            if name == "dp" or all_sizes[name] > 1
        }

    @property
    def heavy_axis_product(self) -> int:
        return self.tp_size * self.cp_size * self.ep_size


@dataclass(frozen=True)
class MeshConstraints:
    input_placements: Optional[list[Optional[tuple[Placement, ...]]]] = None
    output_placements: Optional[list[Optional[tuple[Placement, ...]]]] = None
    parameter_memory_budget: Optional[tuple[Optional[float], Optional[float]]] = None


@dataclass
class MeshEvaluation:
    candidate: MeshCandidate
    feasible: bool
    objective: float = float("inf")
    cost_breakdown: Optional[dict[str, float]] = None
    build_time_s: float = 0.0
    solve_time_s: float = 0.0
    error: Optional[str] = None
    solution: Optional[dict[Any, Any]] = None
    optimizer: Optional[ShardingOptimizer] = None


def _divisors(n: int) -> list[int]:
    result = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            result.append(i)
            if i * i != n:
                result.append(n // i)
    return sorted(result)


def infer_gpus_per_node(default: Optional[int] = None) -> Optional[int]:
    """Infer a canonical node size from the local CUDA device name."""

    try:
        device_name = torch.cuda.get_device_name(0)
    except (RuntimeError, AssertionError):
        return default

    # Keep this ordered from most specific to least specific.
    for gpu_name, gpus_per_node in (
        ("GB200", 72),
        ("B200", 72),
        ("H200", 8),
        ("H100", 8),
        ("A100", 8),
    ):
        if gpu_name in device_name:
            return gpus_per_node
    return default


def generate_dp_tp_mesh_candidates(
    total_gpus: int,
    *,
    gpus_per_node: Optional[int] = None,
    max_tp_size: Optional[int] = None,
    allowed_tp_sizes: Optional[list[int]] = None,
    include_1d: bool = True,
) -> list[MeshCandidate]:
    """Generate topology-aware DP/TP mesh candidates.

    The rightmost axis is TP and is capped to one node by default.  This avoids
    enumerating arbitrary P-way factorizations and keeps solver dimensionality
    bounded by 2 for the no-PP/no-CP/no-EP search mode.
    """

    if total_gpus < 1:
        raise ValueError(f"total_gpus must be positive, got {total_gpus}")

    if gpus_per_node is None:
        gpus_per_node = infer_gpus_per_node(default=total_gpus)
    if gpus_per_node is None:
        gpus_per_node = total_gpus

    tp_cap = min(total_gpus, gpus_per_node)
    if max_tp_size is not None:
        tp_cap = min(tp_cap, max_tp_size)

    if allowed_tp_sizes is None:
        tp_sizes = [d for d in _divisors(total_gpus) if d <= tp_cap]
    else:
        tp_sizes = sorted({t for t in allowed_tp_sizes if t > 0})
        bad = [t for t in tp_sizes if total_gpus % t != 0 or t > tp_cap]
        if bad:
            raise ValueError(
                f"allowed_tp_sizes contains invalid values {bad}; "
                f"total_gpus={total_gpus}, tp_cap={tp_cap}"
            )

    if not include_1d:
        tp_sizes = [t for t in tp_sizes if t != 1]

    candidates: list[MeshCandidate] = []
    for tp_size in tp_sizes:
        dp_size = total_gpus // tp_size
        if tp_size == 1:
            candidates.append(
                MeshCandidate(
                    mesh_shape=(total_gpus,),
                    mesh_dim_names=("dp",),
                    dp_size=total_gpus,
                    tp_size=1,
                    reason="1D DP/FSDP baseline",
                )
            )
        else:
            candidates.append(
                MeshCandidate(
                    mesh_shape=(dp_size, tp_size),
                    mesh_dim_names=("dp", "tp"),
                    dp_size=dp_size,
                    tp_size=tp_size,
                    reason="rightmost TP axis fits within one node",
                )
            )
    return candidates


def generate_semantic_mesh_candidates(
    total_gpus: int,
    *,
    gpus_per_node: Optional[int] = None,
    semantic_axes: tuple[str, ...] = ("tp",),
    axis_order: tuple[str, ...] = ("ep", "cp", "tp"),
    max_ndim: int = 4,
    max_axis_sizes: Optional[dict[str, int]] = None,
    allowed_axis_sizes: Optional[dict[str, list[int]]] = None,
    include_1d: bool = True,
) -> list[MeshCandidate]:
    """Generate exact-solve candidates for semantic mesh roles.

    Non-DP semantic axes are packed into the rightmost, node-local mesh
    dimensions.  This searches higher-rank meshes without enumerating arbitrary
    ordered factorizations: the only factors considered are meaningful roles
    such as TP, CP, and EP.
    """

    if total_gpus < 1:
        raise ValueError(f"total_gpus must be positive, got {total_gpus}")
    if max_ndim < 1:
        raise ValueError(f"max_ndim must be at least 1, got {max_ndim}")

    axes = tuple(dict.fromkeys(axis for axis in semantic_axes if axis != "dp"))
    unknown = sorted(set(axes) - {"tp", "cp", "ep"})
    if unknown:
        raise ValueError(f"Unknown semantic mesh axes: {unknown}")
    unknown_order = sorted(set(axis_order) - {"tp", "cp", "ep"})
    if unknown_order:
        raise ValueError(f"Unknown semantic axis_order entries: {unknown_order}")

    if gpus_per_node is None:
        gpus_per_node = infer_gpus_per_node(default=total_gpus)
    if gpus_per_node is None:
        gpus_per_node = total_gpus

    max_axis_sizes = max_axis_sizes or {}
    allowed_axis_sizes = allowed_axis_sizes or {}
    axis_rank = {axis: idx for idx, axis in enumerate(axis_order)}
    ordered_axes = tuple(
        sorted(axes, key=lambda axis: axis_rank.get(axis, len(axis_rank)))
    )

    candidates: list[MeshCandidate] = []
    seen: set[tuple[tuple[int, ...], tuple[str, ...]]] = set()
    if include_1d:
        baseline = MeshCandidate(
            mesh_shape=(total_gpus,),
            mesh_dim_names=("dp",),
            dp_size=total_gpus,
            reason="1D DP/FSDP baseline",
        )
        candidates.append(baseline)
        seen.add((baseline.mesh_shape, baseline.mesh_dim_names))

    axis_sizes: dict[str, list[int]] = {}
    for axis in ordered_axes:
        axis_cap = min(total_gpus, gpus_per_node, max_axis_sizes.get(axis, total_gpus))
        if axis in allowed_axis_sizes:
            sizes = sorted({size for size in allowed_axis_sizes[axis] if size > 1})
            bad = [s for s in sizes if total_gpus % s != 0 or s > axis_cap]
            if bad:
                raise ValueError(
                    f"allowed_axis_sizes[{axis!r}] contains invalid values {bad}; "
                    f"total_gpus={total_gpus}, axis_cap={axis_cap}"
                )
        else:
            sizes = [d for d in _divisors(total_gpus) if 1 < d <= axis_cap]
        axis_sizes[axis] = sizes

    max_heavy_axes = min(len(ordered_axes), max_ndim - 1)
    for n_axes in range(1, max_heavy_axes + 1):
        for axes_combo in combinations(ordered_axes, n_axes):
            size_lists = [axis_sizes[axis] for axis in axes_combo]
            for sizes_combo in product(*size_lists):
                heavy_product = math.prod(sizes_combo)
                if heavy_product > gpus_per_node or total_gpus % heavy_product != 0:
                    continue

                dp_size = total_gpus // heavy_product
                mesh_shape = (dp_size,) + tuple(sizes_combo)
                mesh_dim_names = ("dp",) + axes_combo
                key = (mesh_shape, mesh_dim_names)
                if key in seen:
                    continue

                kwargs = {
                    "mesh_shape": mesh_shape,
                    "mesh_dim_names": mesh_dim_names,
                    "dp_size": dp_size,
                    "reason": (
                        "semantic heavy axes "
                        + ",".join(
                            f"{axis}={size}"
                            for axis, size in zip(axes_combo, sizes_combo)
                        )
                        + " packed into the rightmost intra-node dimensions"
                    ),
                }
                for axis, size in zip(axes_combo, sizes_combo):
                    if axis == "tp":
                        kwargs["tp_size"] = size
                    elif axis == "cp":
                        kwargs["cp_size"] = size
                    elif axis == "ep":
                        kwargs["ep_size"] = size
                candidates.append(MeshCandidate(**kwargs))
                seen.add(key)

    return candidates


def generate_2d_semantic_mesh_candidates(
    total_gpus: int,
    *,
    gpus_per_node: Optional[int] = None,
    semantic_axes: tuple[str, ...] = ("tp",),
    max_axis_sizes: Optional[dict[str, int]] = None,
    allowed_axis_sizes: Optional[dict[str, list[int]]] = None,
    include_1d: bool = True,
) -> list[MeshCandidate]:
    """Compatibility helper for the first-stage 1D/2D semantic probes."""

    return generate_semantic_mesh_candidates(
        total_gpus,
        gpus_per_node=gpus_per_node,
        semantic_axes=semantic_axes,
        max_ndim=2,
        max_axis_sizes=max_axis_sizes,
        allowed_axis_sizes=allowed_axis_sizes,
        include_1d=include_1d,
    )


def make_axis_placement(
    candidate: MeshCandidate,
    axis_to_tensor_dim: dict[str, int],
) -> tuple[Placement, ...]:
    """Build a placement tuple by mesh-dim name for candidate-specific constraints."""

    return tuple(
        Shard(axis_to_tensor_dim[name]) if name in axis_to_tensor_dim else Replicate()
        for name in candidate.mesh_dim_names
    )


def _iter_tensor_shapes(gm: torch.fx.GraphModule):
    for node in gm.graph.nodes:
        val = node.meta.get("val")
        for leaf in tree_flatten(val)[0]:
            if not isinstance(leaf, torch.Tensor):
                continue
            shape = tuple(leaf.shape)
            if not all(isinstance(dim, int) for dim in shape):
                continue
            if len(shape) == 0:
                continue
            yield shape


def estimate_tp_compatibility_score(gm: torch.fx.GraphModule, tp_size: int) -> float:
    """Shape-only score for whether a TP axis is likely useful.

    This is deliberately model-agnostic: it only asks whether large non-batch
    tensor dimensions can be split by the candidate TP size.  The exact
    placement decision remains with the solver.
    """

    if tp_size <= 1:
        return 0.0

    score = 0.0
    for shape in _iter_tensor_shapes(gm):
        if len(shape) < 2:
            continue
        numel = math.prod(shape)
        weight = math.log2(max(numel, 2))
        non_batch_dims = shape[1:]
        if any(dim >= tp_size and dim % tp_size == 0 for dim in non_batch_dims):
            score += weight
        elif any(dim >= tp_size for dim in non_batch_dims):
            # Uneven shards are valid if non-empty, but they are weaker TP
            # candidates than clean divisors.
            score += 0.25 * weight
        else:
            score -= 0.05 * weight
    return score


def estimate_tp_axis_score(gm: torch.fx.GraphModule, tp_size: int) -> float:
    """Topology-agnostic TP utility with a depth penalty.

    Compatibility alone tends to prefer the largest node-local TP size because
    powers of two divide the same tensor dimensions.  The marginal compute
    benefit from TP saturates as ``1 - 1/tp`` while synchronization depth and
    collective pressure keep growing.  This produces a cheap "elbow" prior; the
    exact solver still evaluates the retained candidates.
    """

    raw = estimate_tp_compatibility_score(gm, tp_size)
    if tp_size <= 1 or raw <= 0:
        return raw
    marginal_compute_benefit = 1.0 - 1.0 / tp_size
    synchronization_depth_penalty = 0.16 * math.log2(tp_size)
    return raw * (marginal_compute_benefit - synchronization_depth_penalty)


def estimate_cp_compatibility_score(gm: torch.fx.GraphModule, cp_size: int) -> float:
    """Shape-only score for a context-parallel axis.

    The heuristic is intentionally generic: it looks for large middle tensor
    dimensions that can be split cleanly.  Transformer sequence dimensions are
    the common case, but the rule does not depend on module names.
    """

    if cp_size <= 1:
        return 0.0

    score = 0.0
    for shape in _iter_tensor_shapes(gm):
        if len(shape) < 3:
            continue
        numel = math.prod(shape)
        weight = math.log2(max(numel, 2))
        middle_dims = shape[1:-1]
        if any(dim >= 128 and dim % cp_size == 0 for dim in middle_dims):
            score += weight
        elif any(dim >= 128 for dim in middle_dims):
            score += 0.25 * weight
    return score


def estimate_ep_compatibility_score(gm: torch.fx.GraphModule, ep_size: int) -> float:
    """Shape-only score for an expert-parallel axis.

    Expert weights commonly appear as rank-3+ tensors with a leading expert
    dimension.  This is only a ranking signal for candidate generation; MoE
    dispatch itself should normally be represented through local_map or custom
    operators with explicit EP boundary placements.
    """

    if ep_size <= 1:
        return 0.0

    score = 0.0
    for shape in _iter_tensor_shapes(gm):
        if len(shape) < 3:
            continue
        expert_dim = shape[0]
        if expert_dim <= 1 or expert_dim > 4096:
            continue
        numel = math.prod(shape)
        weight = math.log2(max(numel, 2))
        if expert_dim % ep_size == 0:
            score += 2.0 * weight
        elif expert_dim >= ep_size:
            score += 0.5 * weight
    return score


def estimate_mesh_candidate_score(
    gm: torch.fx.GraphModule, candidate: MeshCandidate
) -> float:
    """Score a semantic mesh candidate before paying exact optimizer cost."""

    score = 0.0
    score += estimate_tp_axis_score(gm, candidate.tp_size)
    score += estimate_cp_compatibility_score(gm, candidate.cp_size)
    score += estimate_ep_compatibility_score(gm, candidate.ep_size)
    if candidate.ndim > 2:
        score -= 10.0 * (candidate.ndim - 2)
    return score


def rank_mesh_candidates(
    gm: torch.fx.GraphModule,
    candidates: list[MeshCandidate],
    *,
    max_candidates: Optional[int] = None,
) -> list[MeshCandidate]:
    """Rank candidates and optionally keep only the most promising ones.

    The 1D baseline is always retained when present so the search has a
    non-semantic-axis comparison point.
    """

    ranked = [
        replace(
            candidate,
            score=estimate_mesh_candidate_score(gm, candidate),
        )
        for candidate in candidates
    ]
    ranked.sort(key=lambda c: (c.score, c.heavy_axis_product), reverse=True)

    if max_candidates is None or len(ranked) <= max_candidates:
        return ranked

    baseline = [
        c for c in ranked if c.tp_size == 1 and c.cp_size == 1 and c.ep_size == 1
    ]
    non_baseline = [c for c in ranked if c not in baseline]
    keep = max_candidates - len(baseline)
    if keep < 0:
        return baseline[:max_candidates]
    selected = non_baseline[:keep] + baseline
    selected.sort(key=lambda c: (c.score, c.heavy_axis_product), reverse=True)
    return selected


def reset_mesh_search_caches() -> None:
    """Clear caches whose entries depend on mesh shape or topology."""

    reset_placement_options_cache()
    reset_comms_cost_cache()
    reset_compute_cost_cache()


def make_device_mesh(candidate: MeshCandidate, device_type: str = "cuda"):
    with unset_fake_temporarily():
        return init_device_mesh(
            device_type,
            candidate.mesh_shape,
            mesh_dim_names=candidate.mesh_dim_names,
        )


def _set_cost_model_for_mesh(mesh, cost_model: Any) -> None:
    if isinstance(cost_model, NCCLTopoConfig):
        set_nccl_topo_config(cost_model)
    elif cost_model == "nccl":
        set_nccl_topo_config(detect_nccl_topo_config(mesh))
    else:
        set_nccl_topo_config(None)


def _add_parameter_memory_constraint(
    opt: ShardingOptimizer,
    mesh,
    budget: Optional[tuple[Optional[float], Optional[float]]],
) -> None:
    if budget is None:
        return
    low, high = budget
    if low is None:
        low = 0.0
    if high is None:
        high = 1.0 / mesh.size()
    opt.add_parameter_memory_constraint(low, high)


def evaluate_mesh_candidate(
    gm: torch.fx.GraphModule,
    candidate: MeshCandidate,
    constraint_fn: Callable[[MeshCandidate], MeshConstraints],
    *,
    device_type: str = "cuda",
    cost_model: Any = "nccl",
    force_grad_reduce_in_higher_precision: bool = False,
    repeated_subgraphs: bool = True,
    prefetch_discount: Optional[float] = None,
    keep_optimizer: bool = False,
    verbose: bool = False,
) -> MeshEvaluation:
    mesh = make_device_mesh(candidate, device_type=device_type)
    prev_nccl_config = get_nccl_topo_config()

    try:
        _set_cost_model_for_mesh(mesh, cost_model)
        reset_mesh_search_caches()

        t0 = time.perf_counter()
        opt = ShardingOptimizer(
            gm,
            mesh,
            force_grad_reduce_in_higher_precision,
            repeated_subgraphs=repeated_subgraphs,
        )
        constraints = constraint_fn(candidate)
        if constraints.input_placements is not None:
            opt.add_sharded_input_constraint(constraints.input_placements)
        _add_parameter_memory_constraint(opt, mesh, constraints.parameter_memory_budget)
        if constraints.output_placements is not None:
            opt.add_sharded_output_constraint(constraints.output_placements)
        if prefetch_discount is not None:
            opt.apply_prefetch_discount(scale=prefetch_discount)
        t1 = time.perf_counter()

        solution = opt.get_solution(verbose=verbose)
        t2 = time.perf_counter()

        concrete_solution = opt._to_concrete_solution(solution)
        breakdown = opt._compute_solution_cost(concrete_solution)
        objective = pulp.value(opt.prob.objective)
        return MeshEvaluation(
            candidate=candidate,
            feasible=True,
            objective=float(objective),
            cost_breakdown=breakdown,
            build_time_s=t1 - t0,
            solve_time_s=t2 - t1,
            solution=solution,
            optimizer=opt if keep_optimizer else None,
        )
    except Exception as exc:
        logger.exception("Mesh candidate %s failed", candidate.mesh_shape)
        return MeshEvaluation(
            candidate=candidate,
            feasible=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        set_nccl_topo_config(prev_nccl_config)


def search_mesh_candidates(
    gm: torch.fx.GraphModule,
    candidates: list[MeshCandidate],
    constraint_fn: Callable[[MeshCandidate], MeshConstraints],
    **kwargs: Any,
) -> list[MeshEvaluation]:
    """Evaluate mesh candidates in order and keep the best optimizer only."""

    evaluations: list[MeshEvaluation] = []
    best_idx: Optional[int] = None
    best_objective = float("inf")

    for candidate in candidates:
        logger.info(
            "Evaluating mesh candidate shape=%s dim_names=%s tp=%d score=%.1f",
            candidate.mesh_shape,
            candidate.mesh_dim_names,
            candidate.tp_size,
            candidate.score,
        )
        evaluation = evaluate_mesh_candidate(
            gm,
            candidate,
            constraint_fn,
            keep_optimizer=True,
            **kwargs,
        )
        evaluations.append(evaluation)
        if evaluation.feasible:
            logger.info(
                "Mesh candidate %s feasible: objective=%.1f build=%.2fs solve=%.2fs",
                candidate.mesh_shape,
                evaluation.objective,
                evaluation.build_time_s,
                evaluation.solve_time_s,
            )
        else:
            logger.info(
                "Mesh candidate %s infeasible: %s",
                candidate.mesh_shape,
                evaluation.error,
            )
        if evaluation.feasible and evaluation.objective < best_objective:
            if best_idx is not None:
                evaluations[best_idx].optimizer = None
            best_idx = len(evaluations) - 1
            best_objective = evaluation.objective
        elif evaluation.optimizer is not None:
            evaluation.optimizer = None

    return evaluations


def best_mesh_evaluation(evaluations: list[MeshEvaluation]) -> Optional[MeshEvaluation]:
    feasible = [e for e in evaluations if e.feasible]
    if not feasible:
        return None
    return min(feasible, key=lambda e: e.objective)


def format_mesh_evaluations(evaluations: list[MeshEvaluation]) -> str:
    lines = [
        "mesh_shape dim_names axes score objective total compute comm trans build_s solve_s"
    ]
    for e in evaluations:
        c = e.candidate
        axes = ",".join(f"{name}={size}" for name, size in c.semantic_axis_sizes.items())
        if not e.feasible:
            lines.append(
                f"{c.mesh_shape} {c.mesh_dim_names} {axes} "
                f"{c.score:.1f} infeasible error={e.error}"
            )
            continue
        costs = e.cost_breakdown or {}
        lines.append(
            f"{c.mesh_shape} {c.mesh_dim_names} {axes} {c.score:.1f} "
            f"{e.objective:.1f} {costs.get('total', float('nan')):.1f} "
            f"{costs.get('compute', float('nan')):.1f} "
            f"{costs.get('comm', float('nan')):.1f} "
            f"{costs.get('transition', float('nan')):.1f} "
            f"{e.build_time_s:.2f} {e.solve_time_s:.2f}"
        )
    return "\n".join(lines)


def describe_candidate_topology(
    candidate: MeshCandidate, topo_config: NCCLTopoConfig
) -> str:
    lines = [f"mesh_shape={candidate.mesh_shape}, dim_names={candidate.mesh_dim_names}"]
    for idx, name in enumerate(candidate.mesh_dim_names):
        topo = derive_mesh_dim_topo(topo_config, candidate.mesh_shape, idx)
        lines.append(
            f"  dim {idx} ({name}): ranks={topo.n_ranks}, "
            f"nodes={topo.n_nodes}, ppn={topo.ppn}, "
            f"channels={topo.n_channels}"
        )
    return "\n".join(lines)
