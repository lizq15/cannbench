# AdaptiveMaxPool3DGrad CANN Profile Alias Design

## Problem

`aclnnAdaptiveMaxPool3dBackward` may launch either an
`AdaptiveMaxPool3DGrad*` kernel or a `MaxPool3DGradWithArgmax*` kernel,
depending on the installed CANN implementation, target SoC, and input shape.
The operator plugin currently accepts only the adaptive kernel names, so a
valid MaxPool lowering is rejected after successful execution.

The same profile can also contain a `Cast*` kernel produced while converting
indices to the NPU-supported dtype. That setup kernel is not the benchmarked
backward computation and must remain excluded.

## Design

Keep the behavior entirely inside the `adaptivemaxpool3dgrad` operator plugin.
Extend its Ascend `cann_ops_library` profile selection with the
`MaxPool3DGradWithArgmax` pattern while retaining the existing adaptive
patterns. The generic profile parser will continue selecting only names that
match the operator-owned patterns.

Do not enable cross-file aggregation. The CANN dispatch branches are
alternatives: a run launches the adaptive kernel or the MaxPool kernel, not a
two-kernel implementation whose durations should be summed.

## Testing

Add an operator-local regression test that feeds the real parser a synthetic
Ascend profile containing both `Cast*` and `MaxPool3DGradWithArgmax*` rows. The
test must assert that the reported latency comes only from the MaxPool row.
Retain the existing assertions covering adaptive kernel names.

Run the focused operator and profile tests, then the complete `pytest -q`
suite. Rebuild the release and confirm that the staged plugin contains the new
pattern.

## Non-Goals

- Do not modify CLI, backend, config, or generic profile parsing code.
- Do not include `Cast*` in operator latency.
- Do not force all environments to use `MaxPool3DGradWithArgmax`.
- Do not change published result schemas.
