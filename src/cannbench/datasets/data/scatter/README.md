# Scatter Dataset Manifest

This directory contains the built-in `torch.scatter` benchmark datasets used by CannBench.

## Dataset Design

- `smoke`: small rank-2 and rank-3 update fixtures for validation.
- `realistic`: curated model-shaped update cases with TritonBench-style source metadata.
- `stress`: operator-specific boundary cases for long context, wide vocabularies, and large batches.

## Realistic Source Policy

The realistic split follows the same curation policy used for the other indexing/update operators: retain TritonBench-style source metadata, deduplicate by full shape and semantic family, and keep shapes that exercise production model dimensions without copying the full upstream benchmark corpus. Refreshes should record the TritonBench version or commit used to regenerate the manifest.

## Case Tables

### Smoke

| case_id | family | input_shape | index_shape | src_shape | dim | source_model |
| --- | --- | --- | --- | --- | --- | --- |
| `tiny_rank2_scatter` | `token_update` | `[32, 64]` | `[32, 64]` | `[32, 64]` | `1` | `scatter_smoke_fixture` |
| `tiny_rank3_scatter` | `batched_update` | `[4, 16, 32]` | `[4, 16, 32]` | `[4, 16, 32]` | `-1` | `scatter_rank3_fixture` |
| `tiny_sequence_scatter` | `sequence_update` | `[2, 32, 64]` | `[2, 32, 64]` | `[2, 32, 64]` | `1` | `scatter_sequence_fixture` |

### Realistic

| case_id | family | input_shape | index_shape | src_shape | dim | source_model | source_file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bert_token_scatter` | `token_update` | `[16, 128, 30522]` | `[16, 128, 30522]` | `[16, 128, 30522]` | `-1` | `BERT_pytorch` | `torchbench_train/BERT_pytorch_train.json` |
| `t5_attention_scatter` | `attention_update` | `[4, 8, 1024, 1024]` | `[4, 8, 1024, 1024]` | `[4, 8, 1024, 1024]` | `-1` | `T5Small` | `hf_train/T5Small_train.json` |
| `opt_vocab_scatter` | `lm_logits_update` | `[8, 256, 50272]` | `[8, 256, 50272]` | `[8, 256, 50272]` | `-1` | `OPTForCausalLM` | `hf_train/OPTForCausalLM_train.json` |

### Stress

| case_id | family | input_shape | index_shape | src_shape | dim | source_model |
| --- | --- | --- | --- | --- | --- | --- |
| `long_context_scatter` | `long_context_update` | `[2, 16, 4096, 4096]` | `[2, 16, 4096, 4096]` | `[2, 16, 4096, 4096]` | `-1` | `scatter_long_context_boundary` |
| `wide_vocab_scatter` | `wide_vocab_update` | `[4, 512, 128000]` | `[4, 512, 128000]` | `[4, 512, 128000]` | `-1` | `scatter_wide_vocab_boundary` |
| `large_batch_scatter` | `large_batch_update` | `[64, 2048, 512]` | `[64, 2048, 512]` | `[64, 2048, 512]` | `1` | `scatter_large_batch_boundary` |
