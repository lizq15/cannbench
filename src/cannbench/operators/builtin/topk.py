from __future__ import annotations

from cannbench.datasets.materialize import (
    materialize_topk_inputs,
    materialized_values_to_buffer,
)
from cannbench.datasets.topk import get_topk_case, get_topk_dataset
from cannbench.operators.plugin import OperatorPlugin
from cannbench.operators.spec import OperatorSpec


def _build_torch_callable(ctx):
    payload = materialize_topk_inputs(ctx.case, dtype=ctx.request.dtype, seed=ctx.request.seed)
    tensor = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["values"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["input_shape"])
    return lambda: ctx.backend._topk(ctx.torch, tensor, payload)


PLUGIN = OperatorPlugin(
    spec=OperatorSpec(
        name="topk",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="topk",
        runner_name="topk",
    ),
    get_dataset=get_topk_dataset,
    get_case=get_topk_case,
    materialize_inputs=materialize_topk_inputs,
    build_torch_callable=_build_torch_callable,
    sort_order=11,
)
