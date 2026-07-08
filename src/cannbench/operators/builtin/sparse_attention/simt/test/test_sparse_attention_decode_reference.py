from __future__ import annotations

import pytest

from cannbench.operators.builtin.sparse_attention.simt.v1.aten_dsa_sparse_attention import (
    ops,
)


@pytest.mark.parametrize("family", ["family_hd512", "family_hd128"])
def test_sparse_attention_forward_uses_decode_reference_for_decode_fast_path(
    monkeypatch,
    family,
):
    captured: dict[str, object] = {}

    def fake_decode(query, keys, values, indices, *, causal):
        del query, keys, values, indices
        captured["causal"] = causal
        return "decode"

    def unexpected_fallback(query, keys, values, indices, *, causal):
        del query, keys, values, indices, causal
        raise AssertionError("decode fast path should not use fallback reference")

    monkeypatch.setattr(ops, "_decode_reference", fake_decode, raising=False)
    monkeypatch.setattr(ops, "_fallback_reference", unexpected_fallback)

    actual = ops.sparse_attention_forward(
        object(),
        object(),
        object(),
        object(),
        phase="decode",
        family=family,
        causal=True,
    )

    assert actual == "decode"
    assert captured["causal"] is True
