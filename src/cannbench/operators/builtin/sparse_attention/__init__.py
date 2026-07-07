from __future__ import annotations

import math

from cannbench.operators.materialize import materialized_values_to_buffer
from .materialize import materialize_sparse_attention_inputs
from .cases import (
    get_sparse_attention_case,
    get_sparse_attention_dataset,
)
from cannbench.operators.plugin import OperatorPlugin
from cannbench.operators.spec import OperatorSpec
from .external import build_cuda_library_callable, build_vllm_ascend_callable


def _build_torch_callable(ctx):
    payload = materialize_sparse_attention_inputs(
        ctx.case, dtype=ctx.request.dtype, seed=ctx.request.seed
    )
    query = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["query"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["query_shape"])
    keys = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["keys"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["key_shape"])
    values = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["values"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["value_shape"])
    indices = ctx.backend._tensor(
        ctx.torch,
        payload["indices"],
        device=ctx.device,
        dtype=ctx.torch.long,
    ).reshape(payload["indices_shape"])
    def operator():
        batch, query_heads, query_tokens, head_dim = query.shape
        context_tokens = keys.shape[2]
        selected_tokens = indices.shape[2]
        expanded_keys = keys
        expanded_values = values
        if expanded_keys.shape[1] != query_heads:
            repeats = query_heads // expanded_keys.shape[1]
            expanded_keys = expanded_keys.repeat_interleave(repeats, dim=1)
            expanded_values = expanded_values.repeat_interleave(repeats, dim=1)

        gather_index = indices[:, None, :, :, None].expand(
            batch, query_heads, query_tokens, selected_tokens, head_dim
        )
        key_source = expanded_keys[:, :, None, :, :].expand(
            batch, query_heads, query_tokens, context_tokens, head_dim
        )
        value_source = expanded_values[:, :, None, :, :].expand(
            batch, query_heads, query_tokens, context_tokens, head_dim
        )
        selected_keys = ctx.torch.gather(key_source, 3, gather_index)
        selected_values = ctx.torch.gather(value_source, 3, gather_index)
        scores = (query.unsqueeze(3) * selected_keys).sum(dim=-1) / math.sqrt(head_dim)
        if payload["causal"] and payload["phase"] == "prefill":
            positions = ctx.torch.arange(query_tokens, device=query.device).reshape(
                1, 1, query_tokens, 1
            )
            scores = scores.masked_fill(indices[:, None, :, :] > positions, float("-inf"))
        probabilities = ctx.torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
        return (probabilities.unsqueeze(-1) * selected_values).sum(dim=-2)

    return operator


PLUGIN = OperatorPlugin(
    spec=OperatorSpec(
        name="sparse_attention",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="sparse_attention",
        runner_name="sparse_attention",
    ),
    get_dataset=get_sparse_attention_dataset,
    get_case=get_sparse_attention_case,
    materialize_inputs=materialize_sparse_attention_inputs,
    build_torch_callable=_build_torch_callable,
    sort_order=13,
    build_cuda_library_callable=build_cuda_library_callable,
    build_vllm_ascend_callable=build_vllm_ascend_callable,
)
