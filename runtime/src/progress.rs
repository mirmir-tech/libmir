#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressStage {
    LoadWeights,
    PrefillTokens,
    DecodeTokens,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressUnit {
    Byte,
    Token,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgressEvent {
    pub stage: ProgressStage,
    pub current: u64,
    pub total: u64,
    pub unit: ProgressUnit,
    pub detail: String,
}

impl ProgressEvent {
    #[must_use]
    pub fn load_weights(current: u64, total: u64, detail: impl Into<String>) -> Self {
        Self {
            stage: ProgressStage::LoadWeights,
            current,
            total,
            unit: ProgressUnit::Byte,
            detail: detail.into(),
        }
    }

    #[must_use]
    pub fn prefill_tokens(current: usize, total: usize) -> Self {
        Self {
            stage: ProgressStage::PrefillTokens,
            current: current as u64,
            total: total as u64,
            unit: ProgressUnit::Token,
            detail: format!("token {current}/{total}"),
        }
    }

    #[must_use]
    pub fn decode_tokens(current: usize, total: usize) -> Self {
        Self {
            stage: ProgressStage::DecodeTokens,
            current: current as u64,
            total: total as u64,
            unit: ProgressUnit::Token,
            detail: format!("token {current}/{total}"),
        }
    }
}
