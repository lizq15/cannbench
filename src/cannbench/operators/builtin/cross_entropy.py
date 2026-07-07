from __future__ import annotations

from cannbench.datasets.cross_entropy import (
    get_cross_entropy_case,
    get_cross_entropy_dataset,
)
from cannbench.datasets.materialize import (
    materialize_cross_entropy_inputs,
    materialized_values_to_buffer,
)
from cannbench.operators.plugin import OperatorPlugin
from cannbench.operators.spec import OperatorSpec


def _build_torch_callable(ctx):
    payload = materialize_cross_entropy_inputs(
        ctx.case, dtype=ctx.request.dtype, seed=ctx.request.seed
    )
    logits = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["logits"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["logits_shape"])
    targets = ctx.backend._tensor(
        ctx.torch,
        payload["targets"],
        device=ctx.device,
        dtype=ctx.torch.long,
    ).reshape(payload["target_shape"])
    return lambda: ctx.torch.nn.functional.cross_entropy(logits, targets)


PLUGIN = OperatorPlugin(
    spec=OperatorSpec(
        name="cross_entropy",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="cross_entropy",
        runner_name="cross_entropy",
    ),
    get_dataset=get_cross_entropy_dataset,
    get_case=get_cross_entropy_case,
    materialize_inputs=materialize_cross_entropy_inputs,
    build_torch_callable=_build_torch_callable,
    sort_order=8,
)
