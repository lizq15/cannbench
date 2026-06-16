from dataclasses import asdict, dataclass

SUPPORTED_SOFTMAX_DIMS = {0, 1, -1, -2}


@dataclass(frozen=True)
class BenchmarkMetrics:
    iterations: int
    warmup: int
    latency_ms_avg: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    throughput_ops_per_sec: float


@dataclass(frozen=True)
class SoftmaxShape:
    rows: int
    cols: int
    dim: int

    def __post_init__(self) -> None:
        if self.rows <= 0:
            raise ValueError("rows must be > 0")
        if self.cols <= 0:
            raise ValueError("cols must be > 0")
        if self.dim not in SUPPORTED_SOFTMAX_DIMS:
            raise ValueError(
                f"dim must be one of {tuple(sorted(SUPPORTED_SOFTMAX_DIMS))}: {self.dim}"
            )


@dataclass(frozen=True)
class OperatorBenchmarkResult:
    backend: str
    device_name: str
    op: str
    dtype: str
    shape: SoftmaxShape
    metrics: BenchmarkMetrics

    def to_json_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "device_name": self.device_name,
            "op": self.op,
            "dtype": self.dtype,
            "shape": asdict(self.shape),
            "metrics": asdict(self.metrics),
        }
