use serde::Deserialize;

use super::{Family, Reference, TestResult, active_target, require, validation_error};

#[derive(Debug, Deserialize)]
pub struct Float8Reference {
    pub format: String,
    pub scale_mode: String,
    pub scale_granularity: String,
    pub scale_dtype: Option<String>,
    #[serde(default)]
    pub activation_scale: Option<String>,
    #[serde(default)]
    pub input_scale_dtype: Option<String>,
}

impl Reference {
    pub fn validate_float8_for(&self, family: Family) -> TestResult<()> {
        require(self.schema == 2, "FP8 checkpoint reference schema must be 2")?;
        require(self.family == family, "reference semantic family does not match fixture")?;
        require(
            self.affine.is_none()
                && self.packed_int8.is_none()
                && self.packed_int4.is_none()
                && self.awq.is_none()
                && self.gptq.is_none()
                && self.mxfp4.is_none()
                && self.mxfp8.is_none()
                && self.nvfp4.is_none()
                && self.bitsandbytes_4bit.is_none(),
            "FP8 reference contains another compressed storage contract",
        )?;
        let format = self
            .float8
            .as_ref()
            .ok_or_else(|| validation_error("FP8 reference has no format contract"))?;
        let identity = format.scale_mode == "none"
            && format.scale_granularity == "none"
            && format.scale_dtype.is_none();
        let explicit = matches!(format.scale_mode.as_str(), "multiplier" | "inverse_multiplier")
            && matches!(format.scale_granularity.as_str(), "tensor" | "output_channel")
            && format
                .scale_dtype
                .as_deref()
                .is_some_and(|value| matches!(value, "BF16" | "F32"));
        let activation = match format.activation_scale.as_deref() {
            None | Some("dynamic_token") => format.input_scale_dtype.is_none(),
            Some("static_tensor") => format.input_scale_dtype == format.scale_dtype,
            Some(_) => false,
        };
        require(
            matches!(format.format.as_str(), "F8_E4M3" | "F8_E5M2")
                && (identity || explicit)
                && activation,
            "FP8 reference contract is outside checkpoint-test admission",
        )?;
        require(self.vocab_size > 0, "reference vocabulary must not be empty")?;
        self.validate_dtypes()?;
        self.validate_tokens()?;
        Self::validate_logits(&self.first_logits)?;
        self.gate(&active_target())
            .ok_or_else(|| validation_error("FP8 reference has no active-backend gate"))?
            .validate()
    }
}
