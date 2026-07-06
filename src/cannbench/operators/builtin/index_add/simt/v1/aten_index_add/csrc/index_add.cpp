#include <ATen/Functions.h>
#include <ATen/WrapDimUtils.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cstdint>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

extern "C" void launch_index_add_float(
    const int32_t* index,
    const float* source,
    float* output,
    float alpha,
    int64_t index_size,
    int64_t inner_stride,
    int64_t self_dim_size,
    int64_t total_length,
    aclrtStream stream);

extern "C" void launch_index_add_half(
    const int32_t* index,
    const at::Half* source,
    at::Half* output,
    uint16_t alpha,
    int64_t index_size,
    int64_t inner_stride,
    int64_t self_dim_size,
    int64_t total_length,
    aclrtStream stream);

namespace aten_index_add {
namespace {

void run_index_add_float(
    const at::Tensor& index,
    const at::Tensor& source,
    at::Tensor& output,
    float alpha,
    int64_t index_size,
    int64_t inner_stride,
    int64_t self_dim_size,
    int64_t total_length) {
  const auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
  auto acl_call = [=, &output]() -> int {
    launch_index_add_float(
        index.const_data_ptr<int32_t>(),
        source.const_data_ptr<float>(),
        output.mutable_data_ptr<float>(),
        alpha,
        index_size,
        inner_stride,
        self_dim_size,
        total_length,
        acl_stream);
    return 0;
  };
  at_npu::native::OpCommand::RunOpApiV2(
      "aten_index_add::index_add_forward",
      acl_call);
}

void run_index_add_half(
    const at::Tensor& index,
    const at::Tensor& source,
    at::Tensor& output,
    at::Half alpha,
    int64_t index_size,
    int64_t inner_stride,
    int64_t self_dim_size,
    int64_t total_length) {
  const auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
  auto acl_call = [=, &output]() -> int {
    launch_index_add_half(
        index.const_data_ptr<int32_t>(),
        source.const_data_ptr<at::Half>(),
        output.mutable_data_ptr<at::Half>(),
        alpha.x,
        index_size,
        inner_stride,
        self_dim_size,
        total_length,
        acl_stream);
    return 0;
  };
  at_npu::native::OpCommand::RunOpApiV2(
      "aten_index_add::index_add_forward",
      acl_call);
}

at::Tensor index_add_forward_privateuse1(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    const c10::Scalar& alpha) {
  TORCH_CHECK(
      self.device().type() == at::DeviceType::PrivateUse1,
      "expected PrivateUse1/NPU self tensor");
  TORCH_CHECK(
      index.device().type() == at::DeviceType::PrivateUse1,
      "expected PrivateUse1/NPU index tensor");
  TORCH_CHECK(
      source.device().type() == at::DeviceType::PrivateUse1,
      "expected PrivateUse1/NPU source tensor");
  TORCH_CHECK(self.dim() > 0, "index_add self must have at least one dimension");
  TORCH_CHECK(index.dim() == 1, "index_add index must be one-dimensional");
  TORCH_CHECK(
      source.dim() == self.dim(),
      "index_add source rank must match self rank");
  TORCH_CHECK(
      source.scalar_type() == self.scalar_type(),
      "SIMT index_add source dtype must match self dtype");

  const auto wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
  auto source_contiguous = source.contiguous();
  auto index_int = index.to(at::kInt).contiguous();

  const int64_t self_dim_size = self.size(wrapped_dim);
  const int64_t index_size = index_int.numel();
  TORCH_CHECK(
      source_contiguous.size(wrapped_dim) == index_size,
      "index_add source dim size must match index size");
  for (int64_t d = 0; d < self.dim(); ++d) {
    if (d == wrapped_dim) {
      continue;
    }
    TORCH_CHECK(
        source_contiguous.size(d) == self.size(d),
        "index_add source shape must match self outside dim");
  }

  int64_t inner_stride = 1;
  for (int64_t d = wrapped_dim + 1; d < self.dim(); ++d) {
    inner_stride *= self.size(d);
  }

  int64_t outer_size = 1;
  for (int64_t d = 0; d < wrapped_dim; ++d) {
    outer_size *= self.size(d);
  }

  const int64_t total_length = outer_size * index_size * inner_stride;
  if (total_length == 0) {
    return self.contiguous().clone();
  }

  if (self.scalar_type() == at::ScalarType::Float) {
    auto compute_self = self.contiguous();
    auto output = compute_self.clone();
    run_index_add_float(
        index_int,
        source_contiguous,
        output,
        alpha.to<float>(),
        index_size,
        inner_stride,
        self_dim_size,
        total_length);
    return output;
  }

  if (self.scalar_type() == at::ScalarType::Half) {
    auto compute_self = self.contiguous();
    auto output = compute_self.clone();
    run_index_add_half(
        index_int,
        source_contiguous,
        output,
        alpha.to<at::Half>(),
        index_size,
        inner_stride,
        self_dim_size,
        total_length);
    return output;
  }

  auto compute_self = self.contiguous().to(at::kFloat);
  auto compute_source = source_contiguous.to(at::kFloat);
  auto output = compute_self.clone();
  run_index_add_float(
      index_int,
      compute_source,
      output,
      alpha.to<float>(),
      index_size,
      inner_stride,
      self_dim_size,
      total_length);
  return output.to(self.scalar_type());
}

} // namespace

TORCH_LIBRARY_FRAGMENT(aten_index_add, m) {
  m.def(
      "index_add_forward(Tensor self, int dim, Tensor index, Tensor source, Scalar alpha=1) -> Tensor");
}

TORCH_LIBRARY_IMPL(aten_index_add, PrivateUse1, m) {
  m.impl("index_add_forward", &index_add_forward_privateuse1);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::index_add"),
      TORCH_FN(index_add_forward_privateuse1));
}

} // namespace aten_index_add
