use serde::Deserialize;

use super::{Family, Reference, TestResult, active_target, require, validation_error};

#[derive(Debug, Deserialize)]
pub struct MxFp4Reference {
    pub block_size: usize,
    pub storage_dtype: String,
    pub scale_encoding: String,
    pub scale_dtype: String,
    pub output_bias_dtype: String,
    pub routed_layout: String,
}

impl Reference {
    pub fn validate_mxfp4_for(&self, family: Family) -> TestResult<()> {
        require(self.schema == 2, "MXFP4 checkpoint reference schema must be 2")?;
        require(self.family == family, "reference semantic family does not match fixture")?;
        require(
            self.packed_int8.is_none()
                && self.packed_int4.is_none()
                && self.awq.is_none()
                && self.gptq.is_none()
                && self.float8.is_none()
                && self.mxfp8.is_none()
                && self.nvfp4.is_none()
                && self.bitsandbytes_4bit.is_none(),
            "MXFP4 reference contains another compressed storage contract",
        )?;
        let format = self
            .mxfp4
            .as_ref()
            .ok_or_else(|| validation_error("MXFP4 reference has no format contract"))?;
        let interleaved = format.routed_layout == "interleaved_gate_up_bank"
            && format.storage_dtype == "U8"
            && self.affine.is_none();
        let gathered = format.routed_layout == "separate_gate_up_banks"
            && format.storage_dtype == "U32"
            && valid_affine(self.affine.as_ref());
        require(
            format.block_size == 32
                && (interleaved || gathered)
                && format.scale_encoding == "E8M0"
                && format.scale_dtype == "U8"
                && format.output_bias_dtype == "BF16",
            "MXFP4 reference is outside native routed admission",
        )?;
        require(self.vocab_size > 0, "reference vocabulary must not be empty")?;
        self.validate_dtypes()?;
        self.validate_tokens()?;
        Self::validate_logits(&self.first_logits)?;
        self.gate(&active_target())
            .ok_or_else(|| validation_error("MXFP4 reference has no active-backend gate"))?
            .validate()
    }
}

fn valid_affine(affine: Option<&super::AffineReference>) -> bool {
    affine.is_some_and(|format| {
        format.bits == [8]
            && format.group_sizes == [64]
            && matches!(format.parameter_dtype.as_str(), "BF16" | "F16")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_native_routed_reference_contract() -> TestResult<()> {
        let base = include_str!("../../../validation/dense-checkpoint-reference.example.toml")
            .replace("family = \"dense\"", "family = \"clamped_routed\"");
        let source = format!(
            "{base}\n[mxfp4]\nblock_size = 32\nstorage_dtype = \"U8\"\n\
             scale_encoding = \"E8M0\"\nscale_dtype = \"U8\"\n\
             output_bias_dtype = \"BF16\"\nrouted_layout = \"interleaved_gate_up_bank\"\n"
        );
        Reference::parse(&source)?.validate_mxfp4_for(Family::ClampedRouted)
    }

    #[test]
    fn validates_mlx_gathered_reference_contract() -> TestResult<()> {
        let base = include_str!("../../../validation/dense-checkpoint-reference.example.toml")
            .replace("family = \"dense\"", "family = \"shared_routed\"");
        let source = format!(
            "{base}\n[affine]\nbits = [8]\ngroup_sizes = [64]\n\
             parameter_dtype = \"BF16\"\n\n[mxfp4]\nblock_size = 32\n\
             storage_dtype = \"U32\"\nscale_encoding = \"E8M0\"\n\
             scale_dtype = \"U8\"\noutput_bias_dtype = \"BF16\"\n\
             routed_layout = \"separate_gate_up_banks\"\n"
        );
        Reference::parse(&source)?.validate_mxfp4_for(Family::SharedRouted)
    }
}
