# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
from conftest import apply_cuda_patches
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Shard

from autoparallel.api import AutoParallel


class SimpleLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear(x)


class StackedLinear(nn.Module):
    """Multiple identical linear layers for testing repeated_subgraphs."""

    def __init__(self, dim, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(n_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _run_autop(mesh, model_fn, input_fn, mp_policy, repeated_subgraphs=False):
    """Run AutoParallel and return the placement solution."""
    with torch.device("meta"):
        model = model_fn()

    with AutoParallel(
        model, input_fn, mesh, mp_policy, repeated_subgraphs=repeated_subgraphs
    ) as autop:
        autop.add_input_constraints([(Shard(0),) * mesh.ndim])
        autop.add_output_constraints([(Shard(0),) * mesh.ndim])
        sharding_placement = autop.optimize_placement(verbose=False)

    return sharding_placement


def _iter_dtype_casts(sharding_placement, dtype):
    for node, strategy in sharding_placement.items():
        if node.target != torch.ops.autoparallel.dtype_cast.default:
            continue
        if node.meta["val"].dtype == dtype:
            yield node, strategy


def _assert_unary_chain_has_no_pre_cast_redistribution(sharding_placement, cast_node):
    node = cast_node
    while True:
        input_nodes = [n for n in node.all_input_nodes if n in sharding_placement]
        if len(input_nodes) != 1:
            return

        producer = input_nodes[0]
        input_spec = sharding_placement[node].input_specs[0]
        producer_spec = sharding_placement[producer].output_specs
        if not hasattr(input_spec, "placements") or not hasattr(
            producer_spec, "placements"
        ):
            return

        assert input_spec.placements == producer_spec.placements, (
            f"dtype_cast pre-chain edge {producer.name}->{node.name} changes "
            f"placement from {producer_spec.placements} to {input_spec.placements}"
        )
        node = producer


def _assert_reduce_after_cast(sharding_placement, reduce_dtype, min_casts=1):
    matched = 0
    for node, strategy in _iter_dtype_casts(sharding_placement, reduce_dtype):
        output_spec = strategy.output_specs
        if not hasattr(output_spec, "placements") or not any(
            p.is_partial() for p in output_spec.placements
        ):
            continue
        _assert_unary_chain_has_no_pre_cast_redistribution(sharding_placement, node)
        matched += 1

    assert matched >= min_casts, (
        f"Expected at least {min_casts} dtype_cast outputs with Partial placement, "
        f"but found {matched}"
    )


def _assert_forward_allgather_after_cast(sharding_placement, param_dtype, min_casts=1):
    matched = 0
    for node, strategy in _iter_dtype_casts(sharding_placement, param_dtype):
        output_spec = strategy.output_specs
        if not hasattr(output_spec, "placements") or any(
            p.is_partial() for p in output_spec.placements
        ):
            continue
        _assert_unary_chain_has_no_pre_cast_redistribution(sharding_placement, node)
        matched += 1

    assert matched >= min_casts, (
        f"Expected at least {min_casts} forward dtype_cast outputs, "
        f"but found {matched}"
    )


@apply_cuda_patches
def test_grad_reduce_dtype_f32_reduces_after_cast(device_mesh_1d):
    """With reduce_dtype=f32, gradient reductions should happen after dtype_cast."""
    dim = 1024
    mesh = device_mesh_1d

    def model_fn():
        return SimpleLinear(dim)

    def input_fn():
        return torch.randn(2048 * mesh.shape[0], dim, device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    sharding_placement = _run_autop(mesh, model_fn, input_fn, mp_policy)
    _assert_reduce_after_cast(sharding_placement, torch.float32)


@apply_cuda_patches
def test_grad_reduce_dtype_f32_with_repeated_subgraphs(device_mesh_1d):
    """Same as above but with repeated_subgraphs=True (graph clustering).

    Verifies the constraint works correctly when cluster-linked nodes
    copy strategies from representative nodes.
    """
    dim = 1024
    mesh = device_mesh_1d

    def model_fn():
        return StackedLinear(dim, n_layers=4)

    def input_fn():
        return torch.randn(2048 * mesh.shape[0], dim, device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    sharding_placement = _run_autop(
        mesh, model_fn, input_fn, mp_policy, repeated_subgraphs=True
    )
    _assert_reduce_after_cast(sharding_placement, torch.float32, min_casts=2)


@apply_cuda_patches
def test_grad_reduce_dtype_bf16_allows_early_reduction(device_mesh_1d):
    """With reduce_dtype=bf16 (smaller than param_dtype=f32), the constraint
    should NOT fire, and the optimizer is free to reduce before the cast.
    """
    dim = 1024
    mesh = device_mesh_1d

    def model_fn():
        return SimpleLinear(dim)

    def input_fn():
        return torch.randn(2048 * mesh.shape[0], dim, device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32, reduce_dtype=torch.bfloat16
    )
    sharding_placement = _run_autop(mesh, model_fn, input_fn, mp_policy)
    assert sharding_placement is not None, "Optimizer should find a feasible solution"


@apply_cuda_patches
def test_grad_reduce_dtype_same_dtype_no_constraint(device_mesh_1d):
    """With reduce_dtype == param_dtype, no special constraint should fire."""
    dim = 1024
    mesh = device_mesh_1d

    def model_fn():
        return SimpleLinear(dim)

    def input_fn():
        return torch.randn(2048 * mesh.shape[0], dim, device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32, reduce_dtype=torch.float32
    )
    sharding_placement = _run_autop(mesh, model_fn, input_fn, mp_policy)
    assert sharding_placement is not None, "Optimizer should find a feasible solution"


# ---- Forward dtype_cast constraint tests ----


@apply_cuda_patches
def test_fwd_allgather_in_param_dtype(device_mesh_1d):
    """With param_dtype=bf16 and f32 storage, forward allgather should happen
    after the dtype_cast (in bf16), not before it (in f32)."""
    dim = 1024
    mesh = device_mesh_1d

    def model_fn():
        return SimpleLinear(dim)

    def input_fn():
        return torch.randn(2048 * mesh.shape[0], dim, device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    sharding_placement = _run_autop(mesh, model_fn, input_fn, mp_policy)
    _assert_forward_allgather_after_cast(sharding_placement, torch.bfloat16)


@apply_cuda_patches
def test_fwd_allgather_in_param_dtype_reduce_bf16(device_mesh_1d):
    """With param_dtype=bf16, reduce_dtype=bf16, and f32 storage, the forward
    constraint should still fire (it depends on storage > param_dtype, not on
    reduce_dtype)."""
    dim = 1024
    mesh = device_mesh_1d

    def model_fn():
        return SimpleLinear(dim)

    def input_fn():
        return torch.randn(2048 * mesh.shape[0], dim, device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
    )
    sharding_placement = _run_autop(mesh, model_fn, input_fn, mp_policy)
    _assert_forward_allgather_after_cast(sharding_placement, torch.bfloat16)


@apply_cuda_patches
def test_fwd_allgather_with_repeated_subgraphs(device_mesh_1d):
    """Forward constraint with repeated_subgraphs=True (cluster-linked nodes)."""
    dim = 1024
    mesh = device_mesh_1d

    def model_fn():
        return StackedLinear(dim, n_layers=4)

    def input_fn():
        return torch.randn(2048 * mesh.shape[0], dim, device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    sharding_placement = _run_autop(
        mesh, model_fn, input_fn, mp_policy, repeated_subgraphs=True
    )
    _assert_forward_allgather_after_cast(
        sharding_placement, torch.bfloat16, min_casts=2
    )


@apply_cuda_patches
def test_fwd_no_constraint_when_upcasting(device_mesh_1d):
    """With param_dtype=f32 and bf16 storage (upcast), the forward constraint
    should NOT fire since the storage dtype is already smaller."""
    dim = 1024
    mesh = device_mesh_1d

    def model_fn():
        m = SimpleLinear(dim)
        # Force bf16 storage so dtype_cast goes bf16 -> f32
        m.linear.weight = nn.Parameter(m.linear.weight.to(torch.bfloat16))
        return m

    def input_fn():
        return torch.randn(2048 * mesh.shape[0], dim, device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32, reduce_dtype=torch.bfloat16
    )
    sharding_placement = _run_autop(mesh, model_fn, input_fn, mp_policy)
    assert sharding_placement is not None, "Optimizer should find a feasible solution"
