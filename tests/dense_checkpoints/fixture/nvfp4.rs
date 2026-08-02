use serde::Deserialize;

use super::{Family, Reference, TestResult, active_target, require, validation_error};

#[derive(Debug, Deserialize)]
pub struct NvFp4Reference {
    pub block_size: usize,
    pub storage_dtype: String,
    pub scale_encoding: String,
    pub scale_dtype: String,
    pub global_scale_dtype: String,
    pub input_scale_dtype: String,
    pub routed_layout: String,
}

impl Reference {
    pub fn validate_nvfp4_for(&self, family: Family) -> TestResult<()> {
        require(self.schema == 2, "NVFP4 checkpoint reference schema must be 2")?;
        require(self.family == family, "reference semantic family does not match fixture")?;
        require(
            self.affine.is_none()
                && self.packed_int8.is_none()
                && self.packed_int4.is_none()
                && self.awq.is_none()
                && self.gptq.is_none()
                && self.float8.is_none()
                && self.mxfp4.is_none()
                && self.mxfp8.is_none()
                && self.bitsandbytes_4bit.is_none(),
            "NVFP4 reference contains another compressed storage contract",
        )?;
        let format = self
            .nvfp4
            .as_ref()
            .ok_or_else(|| validation_error("NVFP4 reference has no format contract"))?;
        require(
            format.block_size == 16
                && format.storage_dtype == "U8"
                && format.scale_encoding == "F8_E4M3"
                && format.scale_dtype == "F8_E4M3"
                && format.global_scale_dtype == "F32"
                && format.input_scale_dtype == "F32"
                && format.routed_layout == "individual_experts",
            "NVFP4 reference is outside ModelOpt routed admission",
        )?;
        require(self.vocab_size > 0, "reference vocabulary must not be empty")?;
        self.validate_dtypes()?;
        self.validate_tokens()?;
        Self::validate_logits(&self.first_logits)?;
        let gate = self
            .gate(&active_target())
            .ok_or_else(|| validation_error("NVFP4 reference has no active-backend gate"))?;
        if let Some(logits) = &gate.first_logits {
            Self::validate_logits(logits)?;
        }
        gate.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_modelopt_individual_expert_contract() -> TestResult<()> {
        let base = include_str!("../../../validation/dense-checkpoint-reference.example.toml")
            .replace("family = \"dense\"", "family = \"dense_and_routed\"");
        let source = format!(
            "{base}\n[nvfp4]\nblock_size = 16\nstorage_dtype = \"U8\"\n\
             scale_encoding = \"F8_E4M3\"\nscale_dtype = \"F8_E4M3\"\n\
             global_scale_dtype = \"F32\"\ninput_scale_dtype = \"F32\"\n\
             routed_layout = \"individual_experts\"\n"
        );
        Reference::parse(&source)?.validate_nvfp4_for(Family::DenseAndRouted)
    }

    #[test]
    fn validates_pinned_gemma4_reference() -> TestResult<()> {
        Reference::parse(include_str!("../../../validation/references/nvfp4-gemma4-26b-a4b.toml"))?
            .validate_nvfp4_for(Family::DenseAndRouted)
    }
}
