# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the root directory.

import pytest
import torch

from autoparallel.mesh_search import (
    generate_dp_tp_mesh_candidates,
    generate_2d_semantic_mesh_candidates,
    generate_semantic_mesh_candidates,
    make_axis_placement,
    rank_mesh_candidates,
)


def test_generate_dp_tp_mesh_candidates_caps_tp_to_node():
    candidates = generate_dp_tp_mesh_candidates(256, gpus_per_node=8)

    assert [c.mesh_shape for c in candidates] == [
        (256,),
        (128, 2),
        (64, 4),
        (32, 8),
    ]
    assert [c.mesh_dim_names for c in candidates] == [
        ("dp",),
        ("dp", "tp"),
        ("dp", "tp"),
        ("dp", "tp"),
    ]


def test_generate_dp_tp_mesh_candidates_rejects_invalid_allowed_tp():
    with pytest.raises(ValueError, match="invalid values"):
        generate_dp_tp_mesh_candidates(64, gpus_per_node=8, allowed_tp_sizes=[1, 3, 16])


def test_rank_mesh_candidates_keeps_1d_baseline_when_capped():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x

    gm = torch.fx.symbolic_trace(Model())
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            node.meta["val"] = torch.empty(16, 32, device="meta")

    candidates = generate_dp_tp_mesh_candidates(64, gpus_per_node=8)
    ranked = rank_mesh_candidates(gm, candidates, max_candidates=2)

    assert len(ranked) == 2
    assert {c.tp_size for c in ranked} == {1, 4}


def test_generate_semantic_mesh_candidates_emits_2d_role_probes():
    candidates = generate_2d_semantic_mesh_candidates(
        64,
        gpus_per_node=8,
        semantic_axes=("tp", "cp", "ep"),
        max_axis_sizes={"cp": 4},
    )

    assert (64,) in [c.mesh_shape for c in candidates]
    assert (16, 4) in [
        c.mesh_shape for c in candidates if c.mesh_dim_names == ("dp", "cp")
    ]
    assert (8, 8) in [
        c.mesh_shape for c in candidates if c.mesh_dim_names == ("dp", "ep")
    ]
    assert all(c.ndim <= 2 for c in candidates)


def test_generate_semantic_mesh_candidates_rejects_unknown_axis():
    with pytest.raises(ValueError, match="Unknown semantic mesh axes"):
        generate_semantic_mesh_candidates(64, semantic_axes=("tp", "bad_axis"))


def test_generate_semantic_mesh_candidates_emits_node_local_3d_and_4d():
    candidates = generate_semantic_mesh_candidates(
        64,
        gpus_per_node=8,
        semantic_axes=("tp", "cp", "ep"),
        max_axis_sizes={"cp": 4, "ep": 2},
        max_ndim=4,
    )

    assert (8, 2, 4) in [
        c.mesh_shape for c in candidates if c.mesh_dim_names == ("dp", "cp", "tp")
    ]
    assert (8, 2, 2, 2) in [
        c.mesh_shape for c in candidates if c.mesh_dim_names == ("dp", "ep", "cp", "tp")
    ]
    assert all(c.world_size == 64 for c in candidates)
    assert all(c.heavy_axis_product <= 8 for c in candidates)


def test_generate_semantic_mesh_candidates_respects_max_ndim():
    candidates = generate_semantic_mesh_candidates(
        64,
        gpus_per_node=8,
        semantic_axes=("tp", "cp", "ep"),
        max_ndim=3,
    )

    assert all(c.ndim <= 3 for c in candidates)


def test_rank_mesh_candidates_treats_only_all_dp_as_baseline():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x

    gm = torch.fx.symbolic_trace(Model())
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            node.meta["val"] = torch.empty(8, 256, 32, device="meta")

    candidates = generate_semantic_mesh_candidates(
        64,
        gpus_per_node=8,
        semantic_axes=("tp", "cp"),
        max_axis_sizes={"cp": 4},
    )
    ranked = rank_mesh_candidates(gm, candidates, max_candidates=2)

    assert len(ranked) == 2
    assert sum(c.mesh_dim_names == ("dp",) for c in ranked) == 1
    assert any(c.mesh_dim_names != ("dp",) for c in ranked)


def test_make_axis_placement_uses_mesh_dim_names():
    candidates = generate_semantic_mesh_candidates(
        64,
        gpus_per_node=8,
        semantic_axes=("tp", "cp"),
        allowed_axis_sizes={"cp": [2], "tp": [4]},
        max_ndim=3,
        include_1d=False,
    )
    candidate = next(c for c in candidates if c.mesh_dim_names == ("dp", "cp", "tp"))

    placement = make_axis_placement(candidate, {"dp": 0, "cp": 1, "tp": 2})

    assert candidate.mesh_dim_names == ("dp", "cp", "tp")
    assert [str(p) for p in placement] == ["S(0)", "S(1)", "S(2)"]
