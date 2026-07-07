from cannbench.operators import (
    get_operator_plugin,
    list_operator_names,
    list_operator_plugins,
)


def test_operator_plugins_cover_registered_operator_names():
    plugin_names = tuple(plugin.spec.name for plugin in list_operator_plugins())

    assert plugin_names == list_operator_names()


def test_softmax_operator_plugin_owns_dataset_and_materialization():
    plugin = get_operator_plugin("softmax")

    dataset = plugin.get_dataset("smoke")
    case = plugin.get_case("smoke", "tiny_logits")
    payload = plugin.materialize_inputs(case, dtype="float16", seed=7)

    assert plugin.spec.name == "softmax"
    assert dataset.name == "smoke"
    assert case.case_id == "tiny_logits"
    assert payload["shape"] == (32, 128)
    assert payload["dim"] == -1
    assert payload["dtype"] == "float16"
    assert len(payload["values"]) == 32 * 128


def test_embedding_operator_plugin_owns_dataset_and_materialization():
    plugin = get_operator_plugin("embedding")

    dataset = plugin.get_dataset("smoke")
    case = plugin.get_case("smoke", "tiny_token_lookup")
    payload = plugin.materialize_inputs(case, dtype="float16", seed=7)

    assert plugin.spec.name == "embedding"
    assert dataset.name == "smoke"
    assert case.case_id == "tiny_token_lookup"
    assert payload["index_shape"] == (32,)
    assert payload["num_embeddings"] == 128
    assert payload["embedding_dim"] == 64
    assert len(payload["indices"]) == 32
