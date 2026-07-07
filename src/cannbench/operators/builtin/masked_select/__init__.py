from __future__ import annotations

from .cases import (
    get_masked_select_case,
    get_masked_select_dataset,
)
from cannbench.operators.materialize import materialized_values_to_buffer
from .materialize import materialize_masked_select_inputs
from cannbench.operators.plugin import OperatorPlugin
from cannbench.operators.spec import OperatorSpec


def _build_torch_callable(ctx):
    payload = materialize_masked_select_inputs(
        ctx.case, dtype=ctx.request.dtype, seed=ctx.request.seed
    )
    input_tensor = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["values"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["input_shape"])
    mask_tensor = ctx.backend._tensor(
        ctx.torch,
        payload["mask"],
        device=ctx.device,
        dtype=ctx.torch.bool,
    ).reshape(payload["mask_shape"])
    return lambda: ctx.torch.masked_select(input_tensor, mask_tensor)


PLUGIN = OperatorPlugin(
    spec=OperatorSpec(
        name="masked_select",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="masked_select",
        runner_name="masked_select",
    ),
    get_dataset=get_masked_select_dataset,
    get_case=get_masked_select_case,
    materialize_inputs=materialize_masked_select_inputs,
    build_torch_callable=_build_torch_callable,
    sort_order=7,
)
