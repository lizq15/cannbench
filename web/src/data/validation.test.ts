import { describe, expect, it } from "vitest";
import { validateGpuBenchmarkUpload } from "./validation";

const validRecord = {
  schema_version: 1,
  run_id: "gpu-run-1",
  operator: "softmax",
  dataset: "realistic",
  case_id: "gptj_attention",
  family: "attention",
  shape: [1, 16, 128, 128],
  dtype: "float16",
  backend: "nvidia",
  device_class: "H800",
  implementation: "cuda-pytorch",
  implementation_version: "cuda-pytorch",
  source_kind: "real_model",
  source_project: "TritonBench",
  source_model: "GPTJForCausalLM",
  source_file: "hf_train/GPTJForCausalLM_train.json",
  source_op: "aten._softmax.default",
  metrics: { latency_ms: 0.011},
  accuracy: { passed: true, max_abs_error: 0, max_rel_error: 0 },
  diff_ref: null
};

describe("validateGpuBenchmarkUpload", () => {
  it("accepts allowlisted GPU performance records", () => {
    const result = validateGpuBenchmarkUpload({ records: [validRecord] });

    expect(result.ok).toBe(true);
    expect(result.acceptedCount).toBe(1);
  });

  it("accepts GPU CUDA library benchmark records", () => {
    const result = validateGpuBenchmarkUpload({
      records: [
        {
          ...validRecord,
          run_id: "opbench-nvidia-h800-cuda-library-dsa_decode-realistic-bfloat16",
          operator: "dsa_decode",
          dtype: "bfloat16",
          implementation: "cuda_library",
          implementation_version: "cuda-library"
        }
      ]
    });

    expect(result.ok).toBe(true);
    expect(result.acceptedCount).toBe(1);
  });

  it("rejects non-GPU records", () => {
    const result = validateGpuBenchmarkUpload({ records: [{ ...validRecord, backend: "ascend" }] });

    expect(result.ok).toBe(false);
    expect(result.errors).toContain("records[0].backend must be nvidia or gpu");
  });

  it("rejects sensitive fields recursively", () => {
    const result = validateGpuBenchmarkUpload({
      records: [{ ...validRecord, metrics: { ...validRecord.metrics, stdout: "secret" } }]
    });

    expect(result.ok).toBe(false);
    expect(result.errors[0]).toMatch(/sensitive field/i);
  });

  it("rejects unknown top-level fields", () => {
    const result = validateGpuBenchmarkUpload({ records: [{ ...validRecord, hostname: "build-host" }] });

    expect(result.ok).toBe(false);
    expect(result.errors[0]).toMatch(/sensitive field/i);
  });

  it("rejects code snippets embedded in allowed string fields", () => {
    const result = validateGpuBenchmarkUpload({
      records: [{ ...validRecord, implementation_version: "diff --git a/op.cc b/op.cc\n+#include <torch/extension.h>" }]
    });

    expect(result.ok).toBe(false);
    expect(result.errors).toContain("code-like content rejected at payload.records[0].implementation_version");
  });
});
