from __future__ import annotations

import pytest

from cannbench.operators.builtin.lightning_indexer.simt.v1.aten_dsa_lightning_indexer import (
    ops,
)


@pytest.mark.parametrize("family", ["family_64x128", "family_4x64"])
def test_lightning_indexer_forward_uses_decode_reference_for_decode_fast_path(
    monkeypatch,
    family,
):
    captured: dict[str, object] = {}

    def fake_decode(query, keys, weights, *, top_k):
        del query, keys, weights
        captured["top_k"] = top_k
        return "decode"

    def unexpected_fallback(query, keys, weights, *, top_k):
        del query, keys, weights, top_k
        raise AssertionError("decode fast path should not use fallback reference")

    monkeypatch.setattr(ops, "_decode_reference", fake_decode, raising=False)
    monkeypatch.setattr(ops, "_fallback_reference", unexpected_fallback)

    actual = ops.lightning_indexer_forward(
        object(),
        object(),
        object(),
        top_k=4,
        phase="decode",
        family=family,
    )

    assert actual == "decode"
    assert captured["top_k"] == 4
