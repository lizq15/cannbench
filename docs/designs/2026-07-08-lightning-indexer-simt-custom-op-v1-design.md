# Lightning Indexer SIMT Custom-Op V1 Design

## Status

Approved design for the first real Ascend `<<<>>>` custom-op implementation
behind the existing `lightning_indexer` SIMT `v1` interface.

This phase replaces part of the current Python reference-wrapper path with a
real custom-op while preserving the CannBench plugin boundary and fallback
behavior.

## Goal

Build the first real Ascend SIMT custom-op for `lightning_indexer` under the
existing `simt/v1` package, targeting:

- operator: `lightning_indexer`
- phase: `prefill`
- fast-path family: `family_4x64`

Success for this phase means:

- the custom-op builds and installs through the existing operator-local SIMT
  package flow
- `torch.ops.aten_dsa_lightning_indexer.*` is registered and callable
- the `family_4x64 + prefill` path runs through the real custom-op
- output matches the current Python reference wrapper exactly
- all other paths continue to use safe fallback behavior

## Non-Goals

This phase does not include:

- `decode`
- `family_64x128`
- `sparse_attention`
- generic shape support
- performance gates against `vllm-ascend`
- `aclnn`

## Constraints

- Ascend-only scope
- keep all operator-specific logic inside
  `src/cannbench/operators/builtin/lightning_indexer/`
- do not add concrete operator branches to `cli.py`, `core/`, or shared backend
  classes
- continue to use the existing `implementation=simt, implementation_version=v1`
  boundary
- use `softmax`-style custom-op engineering shape:
  `setup.py`, `install.sh`, `scripts/common.sh`, `register.asc`, Python `_C`
  loading, and `torch.ops` dispatch

## Main Decision

Use the existing:

- `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/`

as the first real custom-op home.

`v1` remains the public integration point. The package evolves from:

- Python reference wrapper only

to:

- real custom-op when `_C` is available and the request matches the first
  supported fast path
- Python reference fallback for unsupported shapes, phases, or environments

This avoids duplicating plugin integration in a new `v2` while keeping rollout
risk low.

## Package Structure

The custom-op package should follow the `softmax` engineering shape:

- `aten_dsa_lightning_indexer/__init__.py`
  - load `_C`
  - expose `ops`
- `aten_dsa_lightning_indexer/ops.py`
  - route to real `torch.ops` when available and supported
  - otherwise route to Python reference fallback
- `aten_dsa_lightning_indexer/csrc/register.asc`
  - minimal Python extension entry
- `aten_dsa_lightning_indexer/csrc/*.asc`
  - host-side registration and bridge logic
- `aten_dsa_lightning_indexer/csrc/simt/*.asc`
  - actual `<<<>>>` kernel implementation
- `setup.py`
  - `bisheng -x asc --enable-simt`
- `install.sh`, `scripts/install.sh`, `scripts/common.sh`
  - Ascend environment and install flow

## Public Runtime Interface

The first real custom-op should preserve the existing wrapper boundary:

```python
torch.ops.aten_dsa_lightning_indexer.lightning_indexer_forward(
    query,
    keys,
    weights,
    top_k: int,
    phase: str,
    family: str,
) -> Tensor
```

The Python package continues to expose:

```python
lightning_indexer_forward(query, keys, weights, *, top_k: int, phase: str, family: str)
```

but internally the behavior becomes:

1. if `_C` is loaded, the op is registered, and the request is
   `phase == "prefill"` and `family == "family_4x64"`, call the real custom-op
2. otherwise, use the existing Python reference path

## First Kernel Scope

The first real kernel handles only:

- `phase == "prefill"`
- `family == "family_4x64"`
- logical shape family:
  - `index_heads = 4`
  - `index_dim = 64`

Expected logical computation remains aligned with the current reference:

```python
scores = einsum("bqhd,bcd->bqhc", query, keys)
scores = relu(scores)
scores = scores * weights.unsqueeze(-1)
reduced = scores.sum(dim=2)
indices = topk(reduced, top_k).indices.to(int32)
```

The implementation strategy is not required to mirror PyTorch operator
composition. It only needs to preserve externally visible results.

## Programming Model

This first phase uses the agreed mixed-model direction conservatively:

- the custom-op is implemented through the Ascend SIMT toolchain and `<<<>>>`
  launch path
- only the small first target family is supported
- the purpose is to prove the real custom-op chain, not to finish the full
  cube-plus-SIMT decomposition in one step

The current phase therefore prioritizes:

- real operator registration
- real device execution
- exact correctness against reference

over:

- full shape coverage
- throughput optimization

## Fallback Strategy

Fallback remains mandatory.

The Python wrapper must fall back when:

- `_C` is not built or not importable
- the namespace op is not registered
- `phase != "prefill"`
- `family != "family_4x64"`

Fallback behavior must preserve the current `v1` contract so CannBench runs do
not regress when the real custom-op is unavailable.

## Testing And Acceptance

Acceptance is split into four layers.

### 1. Build And Registration

Prove that:

- the package installs through the operator-local SIMT install flow
- `_C` imports successfully
- `torch.ops.aten_dsa_lightning_indexer.lightning_indexer_forward` exists

### 2. Wrapper Routing

Prove that:

- `prefill + family_4x64` routes to the real custom-op
- unsupported shapes or phases route to Python fallback

### 3. Correctness

Prove that:

- for at least one real `family_4x64` case, the custom-op output matches the
  current Python reference exactly
- output dtype remains `int32`

### 4. CannBench Integration

Prove that:

- `implementation=simt`
- `implementation_version=v1`
- `op=lightning_indexer`

continues to run through the backend and plugin layers successfully.

Full repository verification must remain green with:

```bash
pytest -q
```

## Recommended Execution Order

1. clone the `softmax` SIMT engineering shell into the `lightning_indexer`
   `v1` package shape
2. register a minimal real op under `torch.ops.aten_dsa_lightning_indexer`
3. route `ops.py` to prefer the real op for `prefill + family_4x64`
4. implement a first correct kernel for the `family_4x64` path
5. validate exact correctness and full CannBench integration

## Follow-On Work

If this phase succeeds, the natural next steps are:

1. expand `lightning_indexer` to `family_64x128`
2. add real `decode` support
3. apply the same custom-op engineering shell to `sparse_attention`
4. then optimize toward the mixed cube-plus-SIMT target performance model
