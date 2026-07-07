from __future__ import annotations

import importlib
import pkgutil
from functools import lru_cache

import cannbench.operators.builtin as builtin_operators
from cannbench.operators.plugin import OperatorPlugin
from cannbench.operators.spec import OperatorSpec


@lru_cache(maxsize=1)
def _operator_plugins() -> dict[str, OperatorPlugin]:
    plugins: list[OperatorPlugin] = []
    for module_info in pkgutil.iter_modules(
        builtin_operators.__path__, prefix=f"{builtin_operators.__name__}."
    ):
        if module_info.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        module = importlib.import_module(module_info.name)
        plugin = getattr(module, "PLUGIN", None)
        if plugin is None:
            continue
        if not isinstance(plugin, OperatorPlugin):
            raise TypeError(f"{module_info.name}.PLUGIN must be an OperatorPlugin")
        plugins.append(plugin)

    ordered = sorted(plugins, key=lambda plugin: (plugin.sort_order, plugin.spec.name))
    registry: dict[str, OperatorPlugin] = {}
    for plugin in ordered:
        name = plugin.spec.name
        if name in registry:
            raise ValueError(f"Duplicate operator plugin: {name}")
        registry[name] = plugin
    return registry


def get_operator_plugin(name: str) -> OperatorPlugin:
    try:
        return _operator_plugins()[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported operator: {name}") from exc


def list_operator_plugins() -> tuple[OperatorPlugin, ...]:
    return tuple(_operator_plugins().values())


def get_operator_spec(name: str) -> OperatorSpec:
    return get_operator_plugin(name).spec


def list_operator_names() -> tuple[str, ...]:
    return tuple(_operator_plugins())
