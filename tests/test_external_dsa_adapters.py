import sys
from types import SimpleNamespace

import pytest

from cannbench.core.config import OperatorBenchmarkRequest


def test_operator_request_preserves_external_implementation():
    request = OperatorBenchmarkRequest(
        backend="ascend",
        implementation="vllm_ascend",
        op="lightning_indexer",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_decode_top4",
        warmup=0,
        iterations=1,
    )

    assert request.implementation == "vllm_ascend"


def test_operator_request_rejects_unknown_implementation():
    with pytest.raises(ValueError, match="Unsupported implementation"):
        OperatorBenchmarkRequest(
            backend="ascend",
            implementation="unknown",
            op="lightning_indexer",
            dtype="float16",
            dataset="smoke",
            case_id="tiny_decode_top4",
            warmup=0,
            iterations=1,
        )


def test_ascend_vllm_adapter_calls_torch_npu_lightning_indexer(monkeypatch):
    calls: list[dict[str, object]] = []

    class FakeTensor:
        shape = ()

        def __init__(self, name="tensor"):
            self.name = name

        def reshape(self, *shape):
            self.shape = shape[0] if len(shape) == 1 else shape
            return self

        def to(self, *args, **kwargs):
            return self

    class FakeTorch:
        def __init__(self) -> None:
            self.npu = SimpleNamespace(
                is_available=lambda: True,
                synchronize=lambda: None,
                get_device_name=lambda device: "Fake Ascend",
            )
            self.device = lambda kind: kind
            self.float16 = "float16"
            self.int32 = "int32"
            self.long = "long"
            self.tensor = lambda *args, **kwargs: FakeTensor()

    def fake_lightning_indexer(**kwargs):
        calls.append(kwargs)
        return FakeTensor("indices"), FakeTensor("scores")

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_lightning_indexer=fake_lightning_indexer),
    )

    from cannbench.backends.pytorch_backend import AscendBackend

    request = OperatorBenchmarkRequest(
        backend="ascend",
        implementation="vllm_ascend",
        op="lightning_indexer",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_decode_top4",
        warmup=0,
        iterations=1,
    )

    result = AscendBackend().run_operator(request)

    assert result.backend == "ascend"
    assert calls
    assert calls[0]["layout_query"] == "TND"
    assert calls[0]["layout_key"] == "BSND"
    assert calls[0]["sparse_count"] == 4
    assert calls[0]["sparse_mode"] == 3


def test_ascend_vllm_sparse_attention_calls_sharedkv_metadata_and_op(monkeypatch):
    calls: dict[str, dict[str, object]] = {}

    class FakeTensor:
        def __init__(self, name="tensor", shape=()):
            self.name = name
            self.shape = shape

        def reshape(self, *shape):
            self.shape = shape[0] if len(shape) == 1 else shape
            return self

        def permute(self, *dims):
            self.permuted_dims = dims
            return self

        def contiguous(self):
            return self

    class FakeTorch:
        def __init__(self) -> None:
            self.npu = SimpleNamespace(
                is_available=lambda: True,
                synchronize=lambda: None,
                get_device_name=lambda device: "Fake Ascend",
            )
            self.device = lambda kind: kind
            self.float16 = "float16"
            self.int32 = "int32"
            self.long = "long"
            self.tensor = lambda *args, **kwargs: FakeTensor()
            self.ops = SimpleNamespace(
                _C_ascend=SimpleNamespace(
                    npu_sparse_attn_sharedkv_metadata=self._metadata,
                    npu_sparse_attn_sharedkv=self._attention,
                )
            )

        def _metadata(self, **kwargs):
            calls["metadata"] = kwargs
            return FakeTensor("metadata", (1024,))

        def _attention(self, q, **kwargs):
            calls["attention"] = {"q": q, **kwargs}
            return FakeTensor("out"), FakeTensor("lse")

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())

    from cannbench.backends.pytorch_backend import AscendBackend

    request = OperatorBenchmarkRequest(
        backend="ascend",
        implementation="vllm_ascend",
        op="sparse_attention",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_decode_top4",
        warmup=0,
        iterations=1,
    )

    result = AscendBackend().run_operator(request)

    assert result.backend == "ascend"
    assert calls["metadata"]["num_heads_q"] == 2
    assert calls["metadata"]["num_heads_kv"] == 2
    assert calls["metadata"]["head_dim"] == 16
    assert calls["metadata"]["batch_size"] == 2
    assert calls["metadata"]["max_seqlen_q"] == 1
    assert calls["metadata"]["max_seqlen_kv"] == 32
    assert calls["metadata"]["cmp_topk"] == 4
    assert calls["metadata"]["has_ori_kv"] is False
    assert calls["metadata"]["has_cmp_kv"] is True
    assert calls["attention"]["layout_q"] == "TND"
    assert calls["attention"]["layout_kv"] == "PA_ND"
    assert calls["attention"]["cmp_ratio"] == 1
    assert calls["attention"]["ori_kv"] is None
    assert calls["attention"]["cmp_sparse_indices"].shape == (2, 2, 4)


def test_nvidia_cuda_library_uses_external_lightning_indexer_adapter(monkeypatch):
    calls: list[dict[str, object]] = []

    class FakeTensor:
        def __init__(self, name="tensor", shape=()):
            self.name = name
            self.shape = shape

        def reshape(self, *shape):
            self.shape = shape[0] if len(shape) == 1 else shape
            return self

    class FakeTorch:
        def __init__(self) -> None:
            self.cuda = SimpleNamespace(
                is_available=lambda: True,
                synchronize=lambda: None,
                get_device_name=lambda device: "Fake GPU",
            )
            self.device = lambda kind: kind
            self.float16 = "float16"
            self.int32 = "int32"
            self.tensor = lambda *args, **kwargs: FakeTensor()

    def fake_lightning_indexer(**kwargs):
        calls.append(kwargs)
        return FakeTensor("indices")

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(
        sys.modules,
        "fake_cuda_dsa_adapter",
        SimpleNamespace(lightning_indexer=fake_lightning_indexer),
    )
    monkeypatch.setenv("CANNBENCH_CUDA_DSA_ADAPTER", "fake_cuda_dsa_adapter")

    from cannbench.backends.pytorch_backend import NvidiaBackend

    request = OperatorBenchmarkRequest(
        backend="nvidia",
        implementation="cuda_library",
        op="lightning_indexer",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_decode_top4",
        warmup=0,
        iterations=1,
    )

    result = NvidiaBackend().run_operator(request)

    assert result.backend == "nvidia"
    assert calls
    assert calls[0]["request"] is request
    assert calls[0]["payload"]["top_k"] == 4
    assert calls[0]["query"].shape == (2, 1, 2, 16)
    assert calls[0]["keys"].shape == (2, 32, 16)
    assert calls[0]["weights"].shape == (2, 1, 2)


def test_nvidia_cuda_library_uses_external_sparse_attention_adapter(monkeypatch):
    calls: list[dict[str, object]] = []

    class FakeTensor:
        def __init__(self, name="tensor", shape=()):
            self.name = name
            self.shape = shape

        def reshape(self, *shape):
            self.shape = shape[0] if len(shape) == 1 else shape
            return self

    class FakeTorch:
        def __init__(self) -> None:
            self.cuda = SimpleNamespace(
                is_available=lambda: True,
                synchronize=lambda: None,
                get_device_name=lambda device: "Fake GPU",
            )
            self.device = lambda kind: kind
            self.float16 = "float16"
            self.int32 = "int32"
            self.tensor = lambda *args, **kwargs: FakeTensor()

    def fake_sparse_attention(**kwargs):
        calls.append(kwargs)
        return FakeTensor("out")

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(
        sys.modules,
        "fake_cuda_dsa_adapter",
        SimpleNamespace(sparse_attention=fake_sparse_attention),
    )
    monkeypatch.setenv("CANNBENCH_CUDA_DSA_ADAPTER", "fake_cuda_dsa_adapter")

    from cannbench.backends.pytorch_backend import NvidiaBackend

    request = OperatorBenchmarkRequest(
        backend="nvidia",
        implementation="cuda_library",
        op="sparse_attention",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_decode_top4",
        warmup=0,
        iterations=1,
    )

    result = NvidiaBackend().run_operator(request)

    assert result.backend == "nvidia"
    assert calls
    assert calls[0]["request"] is request
    assert calls[0]["payload"]["phase"] == "decode"
    assert calls[0]["query"].shape == (2, 2, 1, 16)
    assert calls[0]["keys"].shape == (2, 2, 32, 16)
    assert calls[0]["values"].shape == (2, 2, 32, 16)
    assert calls[0]["indices"].shape == (2, 1, 4)


def test_nvidia_cuda_library_dsa_requires_external_adapter(monkeypatch):
    class FakeTorch:
        def __init__(self) -> None:
            self.cuda = SimpleNamespace(is_available=lambda: True)
            self.device = lambda kind: kind
            self.float16 = "float16"
            self.int32 = "int32"
            self.tensor = lambda *args, **kwargs: SimpleNamespace(
                reshape=lambda *shape: None
            )

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.delenv("CANNBENCH_CUDA_DSA_ADAPTER", raising=False)

    from cannbench.backends.pytorch_backend import NvidiaBackend

    request = OperatorBenchmarkRequest(
        backend="nvidia",
        implementation="cuda_library",
        op="sparse_attention",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_decode_top4",
        warmup=0,
        iterations=1,
    )

    with pytest.raises(RuntimeError, match="CANNBENCH_CUDA_DSA_ADAPTER"):
        NvidiaBackend().run_operator(request)


def test_nvidia_cuda_library_rejects_adapter_without_required_callable(monkeypatch):
    class FakeTorch:
        def __init__(self) -> None:
            self.cuda = SimpleNamespace(is_available=lambda: True)
            self.device = lambda kind: kind
            self.float16 = "float16"
            self.int32 = "int32"

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "fake_cuda_dsa_adapter", SimpleNamespace())
    monkeypatch.setenv("CANNBENCH_CUDA_DSA_ADAPTER", "fake_cuda_dsa_adapter")

    from cannbench.backends.pytorch_backend import NvidiaBackend

    request = OperatorBenchmarkRequest(
        backend="nvidia",
        implementation="cuda_library",
        op="sparse_attention",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_decode_top4",
        warmup=0,
        iterations=1,
    )

    with pytest.raises(RuntimeError, match="callable sparse_attention"):
        NvidiaBackend().run_operator(request)
