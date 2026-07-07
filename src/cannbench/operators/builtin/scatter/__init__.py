from __future__ import annotations

from cannbench.operators.materialize import materialized_values_to_buffer
from .materialize import materialize_scatter_inputs
from .cases import get_scatter_case, get_scatter_dataset
from cannbench.operators.plugin import OperatorPlugin
from cannbench.operators.spec import OperatorSpec


def _build_torch_callable(ctx):
    payload = materialize_scatter_inputs(ctx.case, dtype=ctx.request.dtype, seed=ctx.request.seed)
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
    return lambda: ctx.torch.scatter(input_tensor, payload["dim"], index_tensor, src_tensor)


PLUGIN = OperatorPlugin(
    spec=OperatorSpec(
        name="scatter",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="scatter",
        runner_name="scatter",
    ),
    get_dataset=get_scatter_dataset,
    get_case=get_scatter_case,
    materialize_inputs=materialize_scatter_inputs,
    build_torch_callable=_build_torch_callable,
    sort_order=10,
)
