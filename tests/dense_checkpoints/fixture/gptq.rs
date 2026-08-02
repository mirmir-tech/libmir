use super::{Family, Reference, TestResult, active_target, require, validation_error};

impl Reference {
    pub fn validate_gptq_for(&self, family: Family) -> TestResult<()> {
        require(self.schema == 2, "GPTQ checkpoint reference schema must be 2")?;
        require(self.family == family, "reference semantic family does not match fixture")?;
        require(
            self.affine.is_none()
                && self.packed_int8.is_none()
                && self.packed_int4.is_none()
                && self.awq.is_none()
                && self.float8.is_none()
                && self.mxfp4.is_none()
                && self.mxfp8.is_none()
                && self.nvfp4.is_none()
                && self.bitsandbytes_4bit.is_none(),
            "GPTQ reference contains another compressed storage contract",
        )?;
        let format = self
            .gptq
            .as_ref()
            .ok_or_else(|| validation_error("GPTQ reference has no format contract"))?;
        require(
            format.bits == 4
                && format.group_size > 0
                && matches!(format.checkpoint_format.as_str(), "gptq" | "gptq_v2")
                && format.symmetric
                && format.scale_dtype == "F16",
            "GPTQ reference contract is outside initial backend admission",
        )?;
        require(self.vocab_size > 0, "reference vocabulary must not be empty")?;
        self.validate_dtypes()?;
        self.validate_tokens()?;
        Self::validate_logits(&self.first_logits)?;
        self.gate(&active_target())
            .ok_or_else(|| validation_error("GPTQ reference has no active-backend gate"))?
            .validate()
    }
}
