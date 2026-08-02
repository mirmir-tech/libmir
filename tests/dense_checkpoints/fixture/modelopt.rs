use super::{Family, Reference, TestResult, active_target, require, validation_error};

pub fn validate_modelopt_mixed_for(reference: &Reference, family: Family) -> TestResult<()> {
    require(reference.schema == 2, "mixed ModelOpt reference schema must be 2")?;
    require(reference.family == family, "reference semantic family differs")?;
    require(
        reference.affine.is_none()
            && reference.packed_int8.is_none()
            && reference.packed_int4.is_none()
            && reference.awq.is_none()
            && reference.gptq.is_none()
            && reference.mxfp4.is_none()
            && reference.mxfp8.is_none()
            && reference.bitsandbytes_4bit.is_none(),
        "mixed ModelOpt reference contains another packed contract",
    )?;
    let float8 = reference
        .float8
        .as_ref()
        .ok_or_else(|| validation_error("mixed ModelOpt reference has no FP8 contract"))?;
    let nvfp4 = reference
        .nvfp4
        .as_ref()
        .ok_or_else(|| validation_error("mixed ModelOpt reference has no NVFP4 contract"))?;
    require(
        float8.format == "F8_E4M3"
            && float8.scale_mode == "multiplier"
            && float8.scale_granularity == "tensor"
            && float8.scale_dtype.as_deref() == Some("F32")
            && float8.activation_scale.as_deref() == Some("static_tensor")
            && float8.input_scale_dtype.as_deref() == Some("F32"),
        "mixed ModelOpt FP8 contract is unsupported",
    )?;
    require(
        nvfp4.block_size == 16
            && nvfp4.storage_dtype == "U8"
            && nvfp4.scale_encoding == "F8_E4M3"
            && nvfp4.scale_dtype == "F8_E4M3"
            && nvfp4.global_scale_dtype == "F32"
            && nvfp4.input_scale_dtype == "F32"
            && nvfp4.routed_layout == "individual_experts",
        "mixed ModelOpt NVFP4 contract is unsupported",
    )?;
    reference.validate_dtypes()?;
    reference.validate_tokens()?;
    Reference::validate_logits(&reference.first_logits)?;
    reference
        .gate(&active_target())
        .ok_or_else(|| validation_error("mixed ModelOpt reference has no active gate"))?
        .validate()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_pinned_qwen36_reference() -> TestResult<()> {
        let reference = Reference::parse(include_str!(
            "../../../validation/references/modelopt-mixed-qwen36-35b-a3b.toml"
        ))?;
        validate_modelopt_mixed_for(&reference, Family::SharedRouted)?;
        let mut tied = reference.generated_tokens.clone();
        tied[5] = 368;
        let cuda = reference
            .cuda
            .as_ref()
            .ok_or_else(|| validation_error("pinned ModelOpt reference has no CUDA gate"))?;
        assert!(cuda.allows_generation(&tied, &reference.generated_tokens));
        Ok(())
    }
}
