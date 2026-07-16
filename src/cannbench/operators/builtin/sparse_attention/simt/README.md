# sparse_attention SIMT

This directory contains Ascend SIMT integration for `sparse_attention`.

Fast-path families:

- `family_hd512`
- `family_hd128`

Unsupported shapes are rejected by the SIMT plugin path.

## Tensor Shapes

The sparse attention fast path uses the following logical tensor shapes:

- `Q`: `[B, H, Q, D]`
- `K`: `[B, KV_H, C, D]`
- `V`: `[B, KV_H, C, D]`
- `indices`: `[B, Q, S]`
- `output`: `[B, H, Q, D]`
- `lse`: `[B, H, Q]`

Where:

- `B`: batch size
- `H`: query head count
- `KV_H`: key/value head count
- `Q`: query token count
- `C`: context token count
- `S`: selected sparse token count per query token
- `D`: head dimension

In the common decode case, `Q = 1`. In MQA/GQA style layouts, `H` may be larger than `KV_H`, which means multiple query heads share the same KV head group.

## Computation

For a fixed batch `b`, query head `h`, and query token `q`, the operator first uses
`indices[b, q, :]` to choose the sparse context positions:

```text
K_sparse = gather(K, indices)
V_sparse = gather(V, indices)
```

Logical sparse shapes:

- `K_sparse`: `[B, H, Q, S, D]`
- `V_sparse`: `[B, H, Q, S, D]`

Then sparse attention scores are computed only on the selected positions:

```text
scores[b, h, q, s] = dot(Q[b, h, q, :], K_sparse[b, h, q, s, :]) / sqrt(D)
```

This produces:

- `scores`: `[B, H, Q, S]`

If `causal = true`, any selected position whose key token index is greater than the current query token index is masked out before normalization. Invalid sparse indices are also masked out.

The normalized sparse probabilities are:

```text
prob = softmax(scores)
```

with shape:

- `prob`: `[B, H, Q, S]`

The final attention output is the weighted sum over the gathered sparse values:

```text
output[b, h, q, :] = sum_{s in [0, S)} prob[b, h, q, s] * V_sparse[b, h, q, s, :]
```

The operator also returns:

```text
lse[b, h, q] = log(sum(exp(scores[b, h, q, :])))
```

This `lse` term is the per-row log-sum-exp statistic for the sparse attention scores.
