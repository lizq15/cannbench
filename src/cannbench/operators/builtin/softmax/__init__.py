from __future__ import annotations

from cannbench.operators.materialize import materialized_values_to_buffer
from .materialize import materialize_softmax_inputs
from .cases import get_softmax_case, get_softmax_dataset
from cannbench.operators.plugin import OperatorPlugin
from cannbench.operators.spec import OperatorSpec


def _build_torch_callable(ctx):
    payload = materialize_softmax_inputs(ctx.case, dtype=ctx.request.dtype, seed=ctx.request.seed)
    tensor = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["values"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["shape"])
    return lambda: ctx.backend._softmax(ctx.torch, tensor, payload["dim"], ctx.request)


PLUGIN = OperatorPlugin(
    spec=OperatorSpec(
        name="softmax",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="softmax",
        runner_name="softmax",
    ),
    get_dataset=get_softmax_dataset,
    get_case=get_softmax_case,
    materialize_inputs=materialize_softmax_inputs,
    build_torch_callable=_build_torch_callable,
    sort_order=0,
)
