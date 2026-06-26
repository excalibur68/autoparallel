# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the root directory.

import pytest
import torch
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel.cost_models.nccl_cost_model import (
    derive_mesh_dim_topo,
    h100_topo_config,
)
from autoparallel.mesh_search import (
    _factored_seed_cache_key,
    _factored_seed_dim_cost_model,
    generate_2d_semantic_mesh_candidates,
    generate_dp_tp_mesh_candidates,
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


def test_factored_seed_dim_cost_model_preserves_original_mesh_dim_topology():
    config = h100_topo_config(num_nodes=64, gpus_per_node=8)
    mesh_shape = (8, 8, 8)

    for dim_idx in range(3):
        dim_config = _factored_seed_dim_cost_model(
            config, mesh_shape, dim_idx, fabric_aware=True
        )
        original_topo = derive_mesh_dim_topo(config, mesh_shape, dim_idx)
        one_d_topo = derive_mesh_dim_topo(dim_config, (mesh_shape[dim_idx],), 0)

        assert one_d_topo == original_topo

    dim0_topo = derive_mesh_dim_topo(config, mesh_shape, 0)
    dim1_topo = derive_mesh_dim_topo(config, mesh_shape, 1)
    dim2_topo = derive_mesh_dim_topo(config, mesh_shape, 2)

    assert (dim0_topo.n_nodes, dim0_topo.ppn) == (8, 1)
    assert (dim1_topo.n_nodes, dim1_topo.ppn) == (8, 1)
    assert (dim2_topo.n_nodes, dim2_topo.ppn) == (1, 8)


def test_factored_seed_cache_key_separates_same_size_different_fabric():
    config = h100_topo_config(num_nodes=64, gpus_per_node=8)
    mesh_shape = (8, 8, 8)

    rdma_key = _factored_seed_cache_key(
        8, Replicate(), config, mesh_shape, 1, fabric_aware=True
    )
    nvlink_key = _factored_seed_cache_key(
        8, Replicate(), config, mesh_shape, 2, fabric_aware=True
    )
    blind_rdma_key = _factored_seed_cache_key(
        8, Replicate(), config, mesh_shape, 1, fabric_aware=False
    )
    blind_nvlink_key = _factored_seed_cache_key(
        8, Replicate(), config, mesh_shape, 2, fabric_aware=False
    )

    assert rdma_key != nvlink_key
    assert blind_rdma_key == blind_nvlink_key


def test_factored_seed_cache_key_includes_input_placement():
    config = h100_topo_config(num_nodes=64, gpus_per_node=8)
    mesh_shape = (8, 8, 8)

    shard_key = _factored_seed_cache_key(
        8, Shard(0), config, mesh_shape, 0, fabric_aware=True
    )
    replicate_key = _factored_seed_cache_key(
        8, Replicate(), config, mesh_shape, 0, fabric_aware=True
    )

    assert shard_key != replicate_key
