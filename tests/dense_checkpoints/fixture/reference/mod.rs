use std::collections::BTreeSet;

use libmir::BackendTarget;

use super::{Family, LogitsReference, Reference, ResourceGate, TestResult, active_target, require};

mod compressed;

impl Reference {
    pub fn parse(source: &str) -> TestResult<Self> {
        Ok(toml::from_str(source)?)
    }

    pub fn validate_for(&self, family: Family) -> TestResult<()> {
        require(self.schema == 2, "dense checkpoint reference schema must be 2")?;
        require(self.family == family, "reference semantic family does not match catalog")?;
        require(self.affine.is_none(), "dense checkpoint reference contains an affine contract")?;
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
            "dense checkpoint reference contains a packed integer contract",
        )?;
        require(self.vocab_size > 0, "reference vocabulary must not be empty")?;
        self.validate_dtypes()?;
        self.validate_tokens()?;
        Self::validate_logits(&self.first_logits)?;
        if let Some(logits) = self.metal.as_ref().and_then(|gate| gate.first_logits.as_ref()) {
            Self::validate_logits(logits)?;
        }
        if let Some(logits) = self.cuda.as_ref().and_then(|gate| gate.first_logits.as_ref()) {
            Self::validate_logits(logits)?;
        }
        self.gate(&active_target())
            .ok_or_else(|| super::validation_error("reference has no gate for active backend"))?
            .validate()
    }

    pub fn gate(&self, target: &BackendTarget) -> Option<&ResourceGate> {
        match target {
            BackendTarget::Metal => self.metal.as_ref(),
            BackendTarget::Cuda => self.cuda.as_ref(),
            BackendTarget::CpuReference => None,
        }
    }

    pub fn logits(&self, target: &BackendTarget) -> &LogitsReference {
        self.gate(target)
            .and_then(|gate| gate.first_logits.as_ref())
            .unwrap_or(&self.first_logits)
    }

    pub fn tokens(&self, target: &BackendTarget) -> &[u32] {
        self.gate(target)
            .and_then(|gate| gate.generated_tokens.as_deref())
            .unwrap_or(&self.generated_tokens)
    }

    pub(super) fn validate_dtypes(&self) -> TestResult<()> {
        let dtypes = self.dtypes.iter().map(String::as_str).collect::<BTreeSet<_>>();
        require(dtypes.len() == self.dtypes.len(), "reference dtypes must be unique")?;
        require(
            !dtypes.is_empty()
                && dtypes.iter().all(|dtype| matches!(*dtype, "F32" | "F16" | "BF16")),
            "reference must contain only dense F32, F16, or BF16 storage",
        )
    }

    pub(super) fn validate_tokens(&self) -> TestResult<()> {
        require(!self.prompt_tokens.is_empty(), "reference prompt must not be empty")?;
        require(
            self.generated_tokens.len() >= 2,
            "reference generation must contain at least two tokens",
        )?;
        for gate in self.metal.iter().chain(&self.cuda) {
            let tokens = gate.generated_tokens.as_deref().unwrap_or(&self.generated_tokens);
            let logits = gate.first_logits.as_ref().unwrap_or(&self.first_logits);
            require(tokens.len() >= 2, "backend generation must contain at least two tokens")?;
            require(
                logits.token_ids[0] == tokens[0],
                "backend highest reference logit must equal its first greedy token",
            )?;
            for tie in &gate.generated_token_ties {
                require(tie.position < tokens.len(), "generation tie position is out of range")?;
                require(
                    tie.token_ids.contains(&tokens[tie.position]),
                    "generation tie does not contain the canonical token",
                )?;
            }
        }
        require(
            self.first_logits.token_ids[0] == self.generated_tokens[0],
            "highest reference logit must equal the first greedy token",
        )?;
        let vocab = u64::try_from(self.vocab_size)?;
        require(
            self.prompt_tokens
                .iter()
                .chain(&self.generated_tokens)
                .chain(
                    self.metal
                        .iter()
                        .chain(&self.cuda)
                        .filter_map(|gate| gate.generated_tokens.as_ref())
                        .flatten(),
                )
                .chain(&self.first_logits.token_ids)
                .chain(
                    self.metal
                        .iter()
                        .chain(&self.cuda)
                        .filter_map(|gate| gate.first_logits.as_ref())
                        .flat_map(|logits| &logits.token_ids),
                )
                .all(|token| u64::from(*token) < vocab),
            "reference token exceeds checkpoint vocabulary",
        )
    }

    pub(super) fn validate_logits(logits: &LogitsReference) -> TestResult<()> {
        require(
            logits.token_ids.len() == logits.scores.len() && !logits.token_ids.is_empty(),
            "first-logit token and score vectors must be non-empty and equally sized",
        )?;
        require(
            logits.scores.iter().all(|score| score.is_finite())
                && logits.absolute_tolerance.is_finite()
                && logits.absolute_tolerance >= 0.0,
            "first-logit scores and tolerance must be finite and tolerance non-negative",
        )
    }
}
