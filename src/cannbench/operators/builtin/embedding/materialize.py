from __future__ import annotations

import random

from .cases import EmbeddingCase


def materialize_embedding_inputs(
    case: EmbeddingCase, *, dtype: str, seed: int
) -> dict[str, object]:
    generator = random.Random(seed)
    num_indices = 1
    for dim in case.index_shape:
        num_indices *= dim

    indices = tuple(generator.randrange(case.num_embeddings) for _ in range(num_indices))
    weights = tuple(
        round(generator.uniform(-1.0, 1.0), 6)
        for _ in range(case.num_embeddings * case.embedding_dim)
    )
    return {
        "index_shape": case.index_shape,
        "dtype": dtype,
        "indices": indices,
        "weights": weights,
        "num_embeddings": case.num_embeddings,
        "embedding_dim": case.embedding_dim,
    }
