# Masked Select Dataset Manifest

This directory contains the built-in `torch.masked_select` benchmark datasets used by CannBench.

## Dataset Design

- `smoke`: small dense fixtures for validation and CLI flow tests.
- `realistic`: curated real-model masking cases using TritonBench-style workloads and source metadata.
- `stress`: operator-specific boundary cases that stress dense masks, sparse masks, and large activations.

## Case Tables

### Smoke

| case_id | family | input_shape | mask_shape | mask_density | source_model |
| --- | --- | --- | --- | --- | --- |
| `tiny_rank2_masked_select` | `element_filter` | `[32, 64]` | `[32, 64]` | `0.5` | `masked_select_smoke_fixture` |
| `tiny_rank3_masked_select` | `sequence_filter` | `[4, 16, 32]` | `[4, 16, 32]` | `0.4` | `masked_select_rank3_fixture` |
| `tiny_channel_masked_select` | `channel_filter` | `[2, 32, 64]` | `[2, 32, 64]` | `0.3` | `masked_select_channel_fixture` |

### Realistic

| case_id | family | input_shape | mask_shape | mask_density | source_model | source_file |
| --- | --- | --- | --- | --- | --- | --- |
| `bert_attention_masked_scores` | `attention_scores` | `[16, 12, 128, 128]` | `[16, 12, 128, 128]` | `0.75` | `BERT_pytorch` | `torchbench_train/BERT_pytorch_train.json` |
| `t5_attention_masked_scores` | `attention_scores` | `[4, 8, 1024, 1024]` | `[4, 8, 1024, 1024]` | `0.6` | `T5Small` | `hf_train/T5Small_train.json` |
| `opt_logits_masked_filter` | `lm_logits` | `[8, 256, 50272]` | `[8, 256, 50272]` | `0.05` | `OPTForCausalLM` | `hf_train/OPTForCausalLM_train.json` |

### Stress

| case_id | family | input_shape | mask_shape | mask_density | source_model |
| --- | --- | --- | --- | --- | --- |
| `long_context_masked_attention` | `long_context` | `[2, 16, 4096, 4096]` | `[2, 16, 4096, 4096]` | `0.9` | `masked_select_long_context_boundary` |
| `wide_vocab_masked_logits` | `wide_vocab` | `[4, 512, 128000]` | `[4, 512, 128000]` | `0.02` | `masked_select_wide_vocab_boundary` |
| `sparse_batch_filter` | `sparse_filter` | `[64, 2048, 512]` | `[64, 2048, 512]` | `0.1` | `masked_select_sparse_boundary` |
