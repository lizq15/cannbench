from __future__ import annotations

from dataclasses import dataclass

from cannbench.operators import get_operator_plugin


@dataclass(frozen=True)
class OperatorDataset:
    name: str
    dataset_namespace: str

    def get(self, split: str):
        plugin = get_operator_plugin(self.dataset_namespace)
        return plugin.get_dataset(split)


def get_operator_dataset(name: str) -> OperatorDataset:
    plugin = get_operator_plugin(name)
    return OperatorDataset(
        name=plugin.spec.name,
        dataset_namespace=plugin.spec.dataset_namespace,
    )


def get_operator_case(op_name: str, dataset_name: str, case_id: str):
    return get_operator_plugin(op_name).get_case(dataset_name, case_id)
