# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Search semantic mesh shapes for the native full LLaMA3 1B model.

This is a fake-process-group optimizer benchmark: it traces the full model once,
then evaluates topology-aware mesh candidates without applying or running the
resulting parallel graph.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Callable

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel._testing.models.llama3 import (
    Transformer,
    TransformerModelArgs,
    apply_ac,
)
from autoparallel.api import AutoParallel
from autoparallel.cost_models.nccl_cost_model import detect_nccl_topo_config
from autoparallel.mesh_search import (
    MeshCandidate,
    MeshConstraints,
    best_mesh_evaluation,
    describe_candidate_topology,
    format_mesh_evaluations,
    generate_semantic_mesh_candidates,
    make_axis_placement,
    rank_mesh_candidates,
    search_mesh_candidates,
)

logger = logging.getLogger(__name__)


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    return [int(v) for v in value.split(",")]


def _parse_str_tuple(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.split(",") if v.strip())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=64)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument(
        "--semantic-axes",
        type=str,
        default="tp",
        help="Comma-separated non-DP semantic axes to search, e.g. tp,cp,ep",
    )
    parser.add_argument(
        "--axis-order",
        type=str,
        default="ep,cp,tp",
        help="Slow-to-fast order for non-DP axes; rightmost axis is innermost.",
    )
    parser.add_argument(
        "--max-mesh-ndim",
        type=int,
        default=2,
        help="Maximum mesh rank to generate. Use 3 or 4 to include mixed axes.",
    )
    parser.add_argument(
        "--allowed-tp-sizes",
        type=str,
        default=None,
        help="Comma-separated TP sizes to evaluate, e.g. 1,2,4,8",
    )
    parser.add_argument(
        "--allowed-cp-sizes",
        type=str,
        default=None,
        help="Comma-separated CP sizes to evaluate when cp is enabled.",
    )
    parser.add_argument(
        "--allowed-ep-sizes",
        type=str,
        default=None,
        help="Comma-separated EP sizes to evaluate when ep is enabled.",
    )
    parser.add_argument(
        "--prefetch-discount",
        type=float,
        default=0.0,
        help="Scale for prefetchable FSDP communication; 0.0 means fully overlapped.",
    )
    parser.add_argument(
        "--no-prefetch-discount",
        action="store_true",
        help="Disable prefetch discount and use raw communication costs.",
    )
    parser.add_argument(
        "--no-memory-constraint",
        action="store_true",
        help="Do not force parameters to fit within 1/world_size memory.",
    )
    parser.add_argument(
        "--no-vocab-parallel",
        action="store_true",
        help="Keep logits replicated on the TP axis instead of sharding vocab.",
    )
    parser.add_argument(
        "--activation-checkpointing",
        choices=["none", "selective"],
        default="none",
    )
    parser.add_argument(
        "--untied-output",
        action="store_true",
        help="Do not tie the output projection to token embeddings.",
    )
    parser.add_argument("--save-placements", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def llama3_1b_args(seq_len: int) -> TransformerModelArgs:
    # Native approximation of Meta Llama 3.2 1B: full depth/width/vocab.
    # The local Transformer computes the SwiGLU hidden size from
    # ffn_dim_multiplier and multiple_of; these values produce 8192.
    return TransformerModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.5,
        multiple_of=256,
        rope_theta=500000,
        vocab_size=128256,
        max_seq_len=seq_len,
    )


def init_fake_pg(world_size: int) -> None:
    if torch.distributed.is_initialized():
        initialized = torch.distributed.get_world_size()
        if initialized != world_size:
            raise RuntimeError(
                f"Process group already initialized with world_size={initialized}; "
                f"requested {world_size}"
            )
        return
    torch.distributed.init_process_group(
        "fake", store=FakeStore(), rank=0, world_size=world_size
    )


def make_constraint_fn(args) -> Callable[[MeshCandidate], MeshConstraints]:
    def constraints(candidate: MeshCandidate) -> MeshConstraints:
        input_axes = {"dp": 0}
        if candidate.cp_size > 1:
            input_axes["cp"] = 1
        x_sharding = make_axis_placement(candidate, input_axes)

        output_axes = dict(input_axes)
        if candidate.tp_size > 1 and not args.no_vocab_parallel:
            output_axes["tp"] = 2
        out_sharding = make_axis_placement(candidate, output_axes)

        memory = None if args.no_memory_constraint else (None, None)
        return MeshConstraints(
            input_placements=[x_sharding],
            output_placements=[out_sharding],
            parameter_memory_budget=memory,
        )

    return constraints


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    world_size = args.world_size
    batch_size = args.batch_size or world_size
    if batch_size < world_size:
        raise ValueError(
            f"batch_size={batch_size} must be >= world_size={world_size} so "
            "the 1D baseline has non-empty batch shards"
        )

    init_fake_pg(world_size)

    seed_mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("dp",)
    )
    model_args = llama3_1b_args(args.seq_len)

    def model_fn():
        return Transformer(model_args)

    def input_fn():
        return torch.randint(
            0,
            model_args.vocab_size,
            (batch_size, args.seq_len),
            device="cuda",
        )

    with torch.device("meta"):
        model = model_fn()
    if not args.untied_output:
        model.output.weight = model.tok_embeddings.weight
    if args.activation_checkpointing == "selective":
        apply_ac(model, mode="selective", selective_ac_option="op")

    nparams = sum(p.numel() for p in model.parameters())
    print(
        f"Native LLaMA3 1B config: {nparams / 1e9:.2f}B params, "
        f"layers={model_args.n_layers}, dim={model_args.dim}, "
        f"seq_len={args.seq_len}, batch_size={batch_size}"
    )

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    force_grad_reduce = (
        mp_policy.reduce_dtype is not None
        and mp_policy.param_dtype is not None
        and mp_policy.reduce_dtype.itemsize > mp_policy.param_dtype.itemsize
    )

    autop = AutoParallel(
        model,
        input_fn,
        seed_mesh,
        mp_policy=mp_policy,
        repeated_subgraphs=True,
    )

    try:
        t_trace = time.perf_counter()
        autop.build_model_graph()
        print(f"Graph tracing took {time.perf_counter() - t_trace:.2f}s")

        allowed_axis_sizes = {
            axis: sizes
            for axis, sizes in {
                "tp": _parse_int_list(args.allowed_tp_sizes),
                "cp": _parse_int_list(args.allowed_cp_sizes),
                "ep": _parse_int_list(args.allowed_ep_sizes),
            }.items()
            if sizes is not None
        }
        candidates = generate_semantic_mesh_candidates(
            world_size,
            gpus_per_node=args.gpus_per_node,
            semantic_axes=_parse_str_tuple(args.semantic_axes),
            axis_order=_parse_str_tuple(args.axis_order),
            max_ndim=args.max_mesh_ndim,
            allowed_axis_sizes=allowed_axis_sizes,
        )
        candidates = rank_mesh_candidates(
            autop.gm, candidates, max_candidates=args.max_candidates
        )
        print("Candidate order:")
        for candidate in candidates:
            print(
                f"  shape={candidate.mesh_shape}, names={candidate.mesh_dim_names}, "
                f"axes={candidate.semantic_axis_sizes}, score={candidate.score:.1f}, "
                f"reason={candidate.reason}"
            )

        prefetch_discount = (
            None if args.no_prefetch_discount else args.prefetch_discount
        )
        evaluations = search_mesh_candidates(
            autop.gm,
            candidates,
            make_constraint_fn(args),
            cost_model="nccl",
            force_grad_reduce_in_higher_precision=force_grad_reduce,
            repeated_subgraphs=True,
            prefetch_discount=prefetch_discount,
            verbose=args.verbose_solver,
        )

        print("\nSearch results:")
        print(format_mesh_evaluations(evaluations))

        best = best_mesh_evaluation(evaluations)
        if best is None:
            print("No feasible mesh candidate found")
            return 2

        print(
            f"\nBest mesh: shape={best.candidate.mesh_shape}, "
            f"names={best.candidate.mesh_dim_names}, "
            f"objective={best.objective:.1f}"
        )

        if best.optimizer is not None:
            topo_config = detect_nccl_topo_config(best.optimizer.mesh)
            if topo_config is not None:
                print("\nBest mesh topology:")
                print(describe_candidate_topology(best.candidate, topo_config))

            if args.save_placements is not None:
                best.optimizer.save_placements(args.save_placements)
                print(f"Saved placements to {args.save_placements}")

        if args.output_json is not None:
            payload = {
                "model": {
                    "name": "native_llama3_1b",
                    "params": nparams,
                    "layers": model_args.n_layers,
                    "dim": model_args.dim,
                    "n_heads": model_args.n_heads,
                    "n_kv_heads": model_args.n_kv_heads,
                    "seq_len": args.seq_len,
                    "batch_size": batch_size,
                },
                "world_size": world_size,
                "gpus_per_node": args.gpus_per_node,
                "semantic_axes": list(_parse_str_tuple(args.semantic_axes)),
                "axis_order": list(_parse_str_tuple(args.axis_order)),
                "max_mesh_ndim": args.max_mesh_ndim,
                "prefetch_discount": prefetch_discount,
                "evaluations": [
                    {
                        "mesh_shape": list(e.candidate.mesh_shape),
                        "mesh_dim_names": list(e.candidate.mesh_dim_names),
                        "tp_size": e.candidate.tp_size,
                        "cp_size": e.candidate.cp_size,
                        "ep_size": e.candidate.ep_size,
                        "semantic_axis_sizes": e.candidate.semantic_axis_sizes,
                        "heavy_axis_product": e.candidate.heavy_axis_product,
                        "score": e.candidate.score,
                        "feasible": e.feasible,
                        "objective": e.objective,
                        "cost_breakdown": e.cost_breakdown,
                        "build_time_s": e.build_time_s,
                        "solve_time_s": e.solve_time_s,
                        "error": e.error,
                    }
                    for e in evaluations
                ],
                "best_mesh_shape": list(best.candidate.mesh_shape),
                "best_mesh_dim_names": list(best.candidate.mesh_dim_names),
            }
            args.output_json.write_text(json.dumps(payload, indent=2))
            print(f"Saved summary to {args.output_json}")

    finally:
        autop.stack.__exit__(None, None, None)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
