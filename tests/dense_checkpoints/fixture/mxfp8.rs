use serde::Deserialize;

use super::{Family, Reference, TestResult, active_target, require, validation_error};

#[derive(Debug, Deserialize)]
pub struct MxFp8Reference {
    pub block_size: usize,
    pub storage_dtype: String,
    pub scale_encoding: String,
    pub scale_dtype: String,
}

impl Reference {
    pub fn validate_mxfp8_for(&self, family: Family) -> TestResult<()> {
        require(self.schema == 2, "MXFP8 checkpoint reference schema must be 2")?;
        require(self.family == family, "reference semantic family does not match fixture")?;
        require(
            self.affine.is_none()
                && self.packed_int8.is_none()
                && self.packed_int4.is_none()
                && self.awq.is_none()
                && self.gptq.is_none()
                && self.float8.is_none()
                && self.mxfp4.is_none()
                && self.nvfp4.is_none()
                && self.bitsandbytes_4bit.is_none(),
            "MXFP8 reference contains another compressed storage contract",
        )?;
        let format = self
            .mxfp8
            .as_ref()
            .ok_or_else(|| validation_error("MXFP8 reference has no format contract"))?;
        require(
            format.block_size == 32
                && format.storage_dtype == "U32"
                && format.scale_encoding == "E8M0"
                && format.scale_dtype == "U8",
            "MXFP8 reference is outside native MLX admission",
        )?;
        require(self.vocab_size > 0, "reference vocabulary must not be empty")?;
        self.validate_dtypes()?;
        self.validate_tokens()?;
        Self::validate_logits(&self.first_logits)?;
        self.gate(&active_target())
            .ok_or_else(|| validation_error("MXFP8 reference has no active-backend gate"))?
            .validate()
    }
}
