from __future__ import annotations

from .cases import get_index_put_case, get_index_put_dataset
from cannbench.operators.materialize import materialized_values_to_buffer
from .materialize import materialize_index_put_inputs
from cannbench.operators.plugin import OperatorPlugin
from cannbench.operators.spec import OperatorSpec


def _build_torch_callable(ctx):
    payload = materialize_index_put_inputs(ctx.case, dtype=ctx.request.dtype, seed=ctx.request.seed)
    input_tensor = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["values"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["input_shape"])
    index_tensors = tuple(
        ctx.backend._tensor(
            ctx.torch,
            index_values,
            device=ctx.device,
            dtype=ctx.torch.long,
        ).reshape(index_shape)
        for index_values, index_shape in zip(payload["indices"], payload["index_shapes"])
    )
    values_tensor = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["put_values"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["values_shape"])
    return lambda: ctx.torch.index_put(
        input_tensor,
        index_tensors,
        values_tensor,
        accumulate=payload["accumulate"],
    )


PLUGIN = OperatorPlugin(
    spec=OperatorSpec(
        name="index_put",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="index_put",
        runner_name="index_put",
    ),
    get_dataset=get_index_put_dataset,
    get_case=get_index_put_case,
    materialize_inputs=materialize_index_put_inputs,
    build_torch_callable=_build_torch_callable,
    sort_order=5,
)
