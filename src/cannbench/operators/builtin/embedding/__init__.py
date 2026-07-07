from __future__ import annotations

from .cases import get_embedding_case, get_embedding_dataset
from cannbench.operators.materialize import materialized_values_to_buffer
from .materialize import materialize_embedding_inputs
from cannbench.operators.plugin import OperatorPlugin
from cannbench.operators.spec import OperatorSpec


def _build_torch_callable(ctx):
    payload = materialize_embedding_inputs(ctx.case, dtype=ctx.request.dtype, seed=ctx.request.seed)
    weights = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["weights"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["num_embeddings"], payload["embedding_dim"])
    indices = ctx.backend._tensor(
        ctx.torch,
        payload["indices"],
        device=ctx.device,
        dtype=ctx.torch.long,
    ).reshape(payload["index_shape"])
    module = ctx.torch.nn.Embedding(
        payload["num_embeddings"],
        payload["embedding_dim"],
        device=ctx.device,
        dtype=ctx.dtype,
    )
    module.weight = weights
    return lambda: module(indices)


PLUGIN = OperatorPlugin(
    spec=OperatorSpec(
        name="embedding",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="embedding",
        runner_name="embedding",
    ),
    get_dataset=get_embedding_dataset,
    get_case=get_embedding_case,
    materialize_inputs=materialize_embedding_inputs,
    build_torch_callable=_build_torch_callable,
    sort_order=1,
)
