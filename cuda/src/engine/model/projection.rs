use models::weights::{BlockFormat, WeightBindingPlan};

use crate::ProjectionFormat;

pub(super) fn format(bindings: &WeightBindingPlan) -> ProjectionFormat {
    if bindings.affine_group_size().is_some() {
        ProjectionFormat::Affine
    } else if bindings.uses_float8() {
        ProjectionFormat::DirectFp8
    } else if bindings.uses_block_format(BlockFormat::MxFp4) {
        ProjectionFormat::MxFp4
    } else if bindings.uses_block_format(BlockFormat::MxFp8) {
        ProjectionFormat::MxFp8
    } else if bindings.uses_block_format(BlockFormat::NvFp4) {
        ProjectionFormat::NvFp4
    } else if bindings.uses_packed_int8()
        || bindings.uses_packed_int4()
        || bindings.uses_awq()
        || bindings.uses_gptq()
        || bindings.uses_bitsandbytes_4bit()
    {
        ProjectionFormat::PackedInteger
    } else {
        ProjectionFormat::Bf16
    }
}
