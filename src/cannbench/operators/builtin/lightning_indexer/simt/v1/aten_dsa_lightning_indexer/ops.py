from __future__ import annotations

import torch

__all__ = [
    "lightning_indexer_forward",
    "_prefill_reference",
    "_fallback_reference",
]


def lightning_indexer_forward(
    query,
    keys,
    weights,
    *,
    top_k: int,
    phase: str,
    family: str,
):
    if phase == "prefill" and family in {"family_64x128", "family_4x64"}:
        return _prefill_reference(query, keys, weights, top_k=top_k)
    return _fallback_reference(query, keys, weights, top_k=top_k)


def _prefill_reference(query, keys, weights, *, top_k: int):
    scores = torch.einsum("bqhd,bcd->bqhc", query, keys)
    scores = torch.relu(scores)
    scores = scores * weights.unsqueeze(-1)
    reduced = scores.sum(dim=2)
    return torch.topk(
        reduced,
        top_k,
        dim=-1,
        largest=True,
        sorted=True,
    ).indices.to(torch.int32)


def _fallback_reference(query, keys, weights, *, top_k: int):
    return _prefill_reference(query, keys, weights, top_k=top_k)
