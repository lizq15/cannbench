from __future__ import annotations

from cannbench.datasets.index_add import get_index_add_case, get_index_add_dataset
from cannbench.datasets.materialize import (
    materialize_index_add_inputs,
    materialized_values_to_buffer,
)
from cannbench.operators.plugin import OperatorPlugin
from cannbench.operators.spec import OperatorSpec


def _build_torch_callable(ctx):
    payload = materialize_index_add_inputs(ctx.case, dtype=ctx.request.dtype, seed=ctx.request.seed)
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
    src_tensor = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["src"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["src_shape"])
    return lambda: ctx.torch.index_add(input_tensor, payload["dim"], index_tensor, src_tensor)


PLUGIN = OperatorPlugin(
    spec=OperatorSpec(
        name="index_add",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="index_add",
        runner_name="index_add",
    ),
    get_dataset=get_index_add_dataset,
    get_case=get_index_add_case,
    materialize_inputs=materialize_index_add_inputs,
    build_torch_callable=_build_torch_callable,
    sort_order=4,
)
