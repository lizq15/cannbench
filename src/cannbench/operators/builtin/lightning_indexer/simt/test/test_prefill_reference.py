from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cannbench.operators.builtin.lightning_indexer.simt.v1.aten_dsa_lightning_indexer import (
    ops,
)


def test_prefill_reference_matches_torch_topk():
    query = torch.randn(1, 128, 64, 128)
    keys = torch.randn(1, 512, 1, 128)
    weights = torch.rand(1, 128, 64)

    expected = torch.topk(
        (torch.relu(torch.einsum("bqhd,bcd->bqhc", query, keys)) * weights.unsqueeze(-1)).sum(
            dim=2
        ),
        128,
        dim=-1,
        largest=True,
        sorted=True,
    ).indices

    actual = ops._prefill_reference(query, keys, weights, top_k=128)

    assert torch.equal(actual, expected)


def test_lightning_indexer_forward_uses_fallback_reference_outside_prefill_fast_path():
    query = torch.randn(1, 2, 4, 8)
    keys = torch.randn(1, 16, 1, 8)
    weights = torch.rand(1, 2, 4)

    expected = ops._fallback_reference(query, keys, weights, top_k=4)
    actual = ops.lightning_indexer_forward(
        query,
        keys,
        weights,
        top_k=4,
        phase="decode",
        family="family_4x64",
    )

    assert torch.equal(actual, expected)
