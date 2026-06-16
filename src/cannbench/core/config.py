from dataclasses import dataclass, field

from cannbench.core.result import SUPPORTED_SOFTMAX_DIMS

SUPPORTED_DTYPES = {"float32", "float16", "bfloat16"}


@dataclass(frozen=True)
class OperatorBenchmarkRequest:
    backend: str
    op: str
    dtype: str
    rows: int
    cols: int
    dim: int
    warmup: int
    iterations: int
    output_formats: tuple[str, ...] = field(
        default_factory=lambda: ("json", "csv", "md")
    )

    def __post_init__(self) -> None:
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        if self.rows <= 0:
            raise ValueError("rows must be > 0")
        if self.cols <= 0:
            raise ValueError("cols must be > 0")
        if self.dim not in SUPPORTED_SOFTMAX_DIMS:
            raise ValueError(
                f"dim must be one of {tuple(sorted(SUPPORTED_SOFTMAX_DIMS))}: {self.dim}"
            )
