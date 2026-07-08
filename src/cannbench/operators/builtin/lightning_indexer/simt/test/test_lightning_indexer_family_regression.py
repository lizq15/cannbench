from __future__ import annotations

from cannbench.operators.builtin.lightning_indexer import _select_simt_family


def test_a5_and_v4pro_lightning_indexer_cases_map_to_64x128_family():
    assert _select_simt_family(
        {
            "index_heads": 64,
            "index_dim": 128,
            "phase": "prefill",
            "top_k": 1024,
        }
    ) == "family_64x128"


def test_v32_lightning_indexer_cases_map_to_4x64_family():
    assert _select_simt_family(
        {
            "index_heads": 4,
            "index_dim": 64,
            "phase": "decode",
            "top_k": 2048,
        }
    ) == "family_4x64"
