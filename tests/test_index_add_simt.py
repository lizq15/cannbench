from pathlib import Path


SIMT_INDEX_ADD_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "cannbench"
    / "operators"
    / "builtin"
    / "index_add"
    / "simt"
    / "v1"
    / "aten_index_add"
    / "csrc"
)


def test_index_add_simt_v1_has_native_float16_path():
    cpp_source = (SIMT_INDEX_ADD_ROOT / "index_add.cpp").read_text()
    asc_source = (SIMT_INDEX_ADD_ROOT / "simt" / "index_add.asc").read_text()

    assert "launch_index_add_half" in cpp_source
    assert "launch_index_add_half" in asc_source
    assert "self.scalar_type() == at::ScalarType::Half" in cpp_source
    assert "const_data_ptr<at::Half>()" in cpp_source
    assert "mutable_data_ptr<at::Half>()" in cpp_source
    assert "void run_index_add_float(\n    const at::Tensor& index,\n    const at::Tensor& source,\n    at::Tensor& output,\n    float alpha," in cpp_source
    assert "void run_index_add_half(\n    const at::Tensor& index,\n    const at::Tensor& source,\n    at::Tensor& output,\n    at::Half alpha," in cpp_source
    assert '#include "simt_api/asc_fp16.h"' in asc_source
    assert "index_add_kernel<half, uint16_t>" in asc_source
    assert "__hmul(source, __ushort_as_half(alpha))" in asc_source
    assert "index_add_kernel<__fp16, float>" not in asc_source

    half_branch = cpp_source[
        cpp_source.index("self.scalar_type() == at::ScalarType::Half") :
        cpp_source.index("auto compute_self = self.contiguous().to(at::kFloat)")
    ]
    assert ".to(at::kFloat)" not in half_branch
    assert "alpha.to<at::Half>()" in half_branch

    float_branch = cpp_source[
        cpp_source.index("self.scalar_type() == at::ScalarType::Float") :
        cpp_source.index("self.scalar_type() == at::ScalarType::Half")
    ]
    assert "alpha.to<float>()" in float_branch
