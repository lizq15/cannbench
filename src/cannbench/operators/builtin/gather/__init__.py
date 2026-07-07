from __future__ import annotations

from .cases import get_gather_case, get_gather_dataset
from cannbench.operators.materialize import materialized_values_to_buffer
from .materialize import materialize_gather_inputs
from cannbench.operators.plugin import OperatorPlugin
from cannbench.operators.spec import OperatorSpec


def _build_torch_callable(ctx):
    payload = materialize_gather_inputs(ctx.case, dtype=ctx.request.dtype, seed=ctx.request.seed)
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
    return lambda: ctx.torch.gather(input_tensor, payload["dim"], index_tensor)


PLUGIN = OperatorPlugin(
    spec=OperatorSpec(
        name="gather",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="gather",
        runner_name="gather",
    ),
    get_dataset=get_gather_dataset,
    get_case=get_gather_case,
    materialize_inputs=materialize_gather_inputs,
    build_torch_callable=_build_torch_callable,
    sort_order=2,
)
