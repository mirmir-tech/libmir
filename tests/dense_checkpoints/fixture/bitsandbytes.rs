use models::weights::{
    BitsAndBytes4BitQuantization, BitsAndBytes4BitType, BitsAndBytesComputeDType,
    BitsAndBytesStorageDType,
};
use serde::Deserialize;

use super::{Family, Reference, TestResult, active_target, require, validation_error};

#[derive(Debug, Deserialize)]
pub struct BitsAndBytes4BitReference {
    pub quant_type: BitsAndBytes4BitType,
    pub block_size: usize,
    pub compute_dtype: BitsAndBytesComputeDType,
    pub storage_dtype: BitsAndBytesStorageDType,
    #[serde(default)]
    pub nested_block_size: Option<usize>,
}

impl BitsAndBytes4BitReference {
    pub const fn format(&self) -> BitsAndBytes4BitQuantization {
        BitsAndBytes4BitQuantization {
            quant_type: self.quant_type,
            block_size: self.block_size,
            compute_dtype: self.compute_dtype,
            storage_dtype: self.storage_dtype,
            nested_block_size: self.nested_block_size,
        }
    }
}

impl Reference {
    pub fn validate_bitsandbytes_for(&self, family: Family) -> TestResult<()> {
        require(self.schema == 2, "bitsandbytes checkpoint reference schema must be 2")?;
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
                && self.nvfp4.is_none(),
            "bitsandbytes reference contains another compressed storage contract",
        )?;
        let format = self.bitsandbytes_4bit.as_ref().ok_or_else(|| {
            validation_error("bitsandbytes reference has no 4-bit format contract")
        })?;
        require(format.format().is_supported(), "bitsandbytes format is outside MF-140")?;
        require(self.vocab_size > 0, "reference vocabulary must not be empty")?;
        self.validate_dtypes()?;
        self.validate_tokens()?;
        Self::validate_logits(&self.first_logits)?;
        let gate = self
            .gate(&active_target())
            .ok_or_else(|| validation_error("bitsandbytes reference has no active-backend gate"))?;
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
    fn accepts_direct_fp4_and_nested_nf4_contracts() -> TestResult<()> {
        for section in [
            "quant_type = \"fp4\"\nblock_size = 64\ncompute_dtype = \"bf16\"\n\
             storage_dtype = \"u8\"\n",
            "quant_type = \"nf4\"\nblock_size = 64\ncompute_dtype = \"bf16\"\n\
             storage_dtype = \"bf16\"\nnested_block_size = 256\n",
        ] {
            let source = format!(
                "{}\n[bitsandbytes_4bit]\n{section}",
                include_str!("../../../validation/dense-checkpoint-reference.example.toml")
            );
            Reference::parse(&source)?.validate_bitsandbytes_for(Family::Dense)?;
        }
        Ok(())
    }

    #[test]
    fn validates_pinned_mf140_references() -> TestResult<()> {
        Reference::parse(include_str!(
            "../../../validation/references/bitsandbytes-nf4-smollm2-135m.toml"
        ))?
        .validate_bitsandbytes_for(Family::Dense)?;
        Reference::parse(include_str!(
            "../../../validation/references/bitsandbytes-fp4-qwen3.toml"
        ))?
        .validate_bitsandbytes_for(Family::Dense)
    }
}
