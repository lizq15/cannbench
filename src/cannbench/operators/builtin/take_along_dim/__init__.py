from __future__ import annotations

from cannbench.operators.materialize import materialized_values_to_buffer
from .materialize import materialize_take_along_dim_inputs
from .cases import (
    get_take_along_dim_case,
    get_take_along_dim_dataset,
)
from cannbench.operators.plugin import OperatorPlugin
from cannbench.operators.spec import OperatorSpec


def _build_torch_callable(ctx):
    payload = materialize_take_along_dim_inputs(
        ctx.case, dtype=ctx.request.dtype, seed=ctx.request.seed
    )
    input_tensor = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["values"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["input_shape"])
    index_tensor = ctx.backend._tensor(
        ctx.torch,
        payload["indices"],
        device=ctx.device,
        dtype=ctx.torch.long,
    ).reshape(payload["index_shape"])
    return lambda: ctx.torch.take_along_dim(input_tensor, index_tensor, payload["dim"])


PLUGIN = OperatorPlugin(
    spec=OperatorSpec(
        name="take_along_dim",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="take_along_dim",
        runner_name="take_along_dim",
    ),
    get_dataset=get_take_along_dim_dataset,
    get_case=get_take_along_dim_case,
    materialize_inputs=materialize_take_along_dim_inputs,
    build_torch_callable=_build_torch_callable,
    sort_order=6,
)
