# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
from conftest import apply_cuda_patches
from torch import nn
from torch.distributed.tensor.placement_types import Shard

from autoparallel.api import AutoParallel


class _RepeatedLayerModel(nn.Module):
    def __init__(self, dim, n_layers):
        super().__init__()
        self.embed = nn.Linear(dim, dim, bias=False)
        self.layers = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(n_layers)]
        )
        self.head = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def _setup_autop(model, dim, device_mesh):
    batch_size = 2 * device_mesh.shape[0]

    def input_fn():
        return torch.rand(batch_size, dim, device="cuda")

    return AutoParallel(model, input_fn, device_mesh, repeated_subgraphs=True)


@apply_cuda_patches
def test_export_json_handles_repeated_layer_clusters(device_mesh_1d):
    dim = 64
    with torch.device("meta"):
        model = _RepeatedLayerModel(dim, n_layers=4)

    autop = _setup_autop(model, dim, device_mesh_1d)
    with autop:
        autop.add_input_constraints([(Shard(0),)])
        autop.add_output_constraints([(Shard(0),)])
        autop.sharding_optimizer.get_solution()
        data = autop.sharding_optimizer.get_json()

    assert data["nodes"]
    assert any("cluster_id" in node for node in data["nodes"])


# ---- export_sharding_json tests ----


@apply_cuda_patches
def test_export_json_produces_valid_structure(device_mesh_1d):
    dim = 64
    with torch.device("meta"):
        model = _RepeatedLayerModel(dim, n_layers=2)

    autop = _setup_autop(model, dim, device_mesh_1d)
    with autop:
        autop.add_input_constraints([(Shard(0),)])
        autop.add_output_constraints([(Shard(0),)])
        autop.sharding_optimizer.get_solution()
        data = autop.sharding_optimizer.get_json()

    assert "nodes" in data
    assert "mesh" in data
    assert "summary" in data
    assert isinstance(data["nodes"], list)
    assert len(data["nodes"]) > 0

    # Every node should have required fields
    for node in data["nodes"]:
        assert "name" in node
        assert "op" in node

    # Summary should have cost fields
    assert "total" in data["summary"]
    assert "comm" in data["summary"]
    assert "compute" in data["summary"]

    # Mesh should have shape
    assert "shape" in data["mesh"]
