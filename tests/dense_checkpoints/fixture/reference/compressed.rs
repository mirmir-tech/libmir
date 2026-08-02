use std::collections::BTreeSet;

use super::*;
use crate::fixture::validation_error;

impl Reference {
    pub fn validate_affine_for(&self, family: Family) -> TestResult<()> {
        require(self.schema == 2, "affine checkpoint reference schema must be 2")?;
        require(self.family == family, "reference semantic family does not match fixture")?;
        require(self.vocab_size > 0, "reference vocabulary must not be empty")?;
        require(
            self.packed_int8.is_none()
                && self.packed_int4.is_none()
                && self.awq.is_none()
                && self.gptq.is_none()
                && self.float8.is_none()
                && self.mxfp4.is_none()
                && self.mxfp8.is_none()
                && self.nvfp4.is_none()
                && self.bitsandbytes_4bit.is_none(),
            "affine checkpoint reference contains a packed integer contract",
        )?;
        self.validate_dtypes()?;
        self.validate_affine()?;
        self.validate_tokens()?;
        Self::validate_logits(&self.first_logits)?;
        for logits in self
            .metal
            .iter()
            .chain(&self.cuda)
            .filter_map(|gate| gate.first_logits.as_ref())
        {
            Self::validate_logits(logits)?;
        }
        self.gate(&active_target())
            .ok_or_else(|| validation_error("reference has no gate for active backend"))?
            .validate()
    }

    pub fn validate_packed_int8_for(&self, family: Family) -> TestResult<()> {
        require(
            self.packed_int4.is_none()
                && self.awq.is_none()
                && self.gptq.is_none()
                && self.float8.is_none()
                && self.mxfp4.is_none()
                && self.mxfp8.is_none()
                && self.nvfp4.is_none()
                && self.bitsandbytes_4bit.is_none(),
            "packed INT8 reference contains another integer contract",
        )?;
        self.validate_packed_integer_for(family, 8)
    }

    pub fn validate_packed_int4_for(&self, family: Family) -> TestResult<()> {
        require(
            self.packed_int8.is_none()
                && self.awq.is_none()
                && self.gptq.is_none()
                && self.float8.is_none()
                && self.mxfp4.is_none()
                && self.mxfp8.is_none()
                && self.nvfp4.is_none()
                && self.bitsandbytes_4bit.is_none(),
            "packed INT4 reference contains another integer contract",
        )?;
        self.validate_packed_integer_for(family, 4)
    }

    pub fn validate_awq_for(&self, family: Family) -> TestResult<()> {
        require(self.schema == 2, "AWQ checkpoint reference schema must be 2")?;
        require(self.family == family, "reference semantic family does not match fixture")?;
        require(self.affine.is_none(), "AWQ reference contains affine storage")?;
        require(
            self.packed_int8.is_none()
                && self.packed_int4.is_none()
                && self.gptq.is_none()
                && self.float8.is_none()
                && self.mxfp4.is_none()
                && self.mxfp8.is_none()
                && self.nvfp4.is_none()
                && self.bitsandbytes_4bit.is_none(),
            "AWQ reference contains compressed integer storage",
        )?;
        let awq = self
            .awq
            .as_ref()
            .ok_or_else(|| validation_error("AWQ reference has no format contract"))?;
        require(awq.bits == 4 && awq.group_size > 0, "AWQ reference contract is invalid")?;
        require(self.vocab_size > 0, "reference vocabulary must not be empty")?;
        self.validate_dtypes()?;
        self.validate_tokens()?;
        Self::validate_logits(&self.first_logits)?;
        self.gate(&active_target())
            .ok_or_else(|| validation_error("AWQ reference has no active-backend gate"))?
            .validate()
    }

    fn validate_affine(&self) -> TestResult<()> {
        let affine = self
            .affine
            .as_ref()
            .ok_or_else(|| validation_error("affine reference has no storage contract"))?;
        let bits = affine.bits.iter().copied().collect::<BTreeSet<_>>();
        let groups = affine.group_sizes.iter().copied().collect::<BTreeSet<_>>();
        require(
            bits.len() == affine.bits.len()
                && !bits.is_empty()
                && bits.iter().all(|bits| matches!(bits, 2 | 3 | 4 | 5 | 6 | 8)),
            "affine reference has invalid or duplicate bit widths",
        )?;
        require(
            groups.len() == affine.group_sizes.len()
                && !groups.is_empty()
                && groups.iter().all(|group| *group > 0),
            "affine reference has invalid or duplicate group sizes",
        )?;
        require(
            matches!(affine.parameter_dtype.as_str(), "F16" | "BF16" | "F32"),
            "affine reference has an unsupported parameter dtype",
        )
    }

    fn validate_packed_integer_for(&self, family: Family, bits: u8) -> TestResult<()> {
        require(self.schema == 2, "packed integer reference schema must be 2")?;
        require(self.family == family, "reference semantic family does not match fixture")?;
        require(self.affine.is_none(), "packed integer reference contains affine storage")?;
        require(self.vocab_size > 0, "reference vocabulary must not be empty")?;
        self.validate_dtypes()?;
        self.validate_packed_integer(bits)?;
        self.validate_tokens()?;
        Self::validate_logits(&self.first_logits)?;
        self.gate(&active_target())
            .ok_or_else(|| validation_error("packed reference has no active-backend gate"))?
            .validate()
    }

    fn validate_packed_integer(&self, bits: u8) -> TestResult<()> {
        let format = match bits {
            4 => self.packed_int4.as_ref(),
            8 => self.packed_int8.as_ref(),
            _ => None,
        }
        .ok_or_else(|| validation_error("reference has no packed integer contract"))?;
        let expected_strategy = if bits == 4 {
            "group"
        } else {
            "channel"
        };
        require(
            format.bits == bits
                && format.scale_strategy == expected_strategy
                && format.group_size.is_some() == (bits == 4)
                && format.group_size.is_none_or(|size| size > 0)
                && format.signedness == "offset_binary"
                && format.zero_point == "none"
                && format.activation_order == "none"
                && format.packing == "dense_little_endian"
                && format.storage_dtype == "I32"
                && format.scale_dtype == "BF16",
            "packed integer reference is not the admitted contract",
        )
    }
}
