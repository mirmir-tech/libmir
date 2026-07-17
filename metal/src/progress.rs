#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalProgressStage {
    LoadWeights,
    PrefillTokens,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalProgressUnit {
    Byte,
    Token,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalProgressEvent {
    pub stage: MetalProgressStage,
    pub current: u64,
    pub total: u64,
    pub unit: MetalProgressUnit,
    pub detail: String,
}

impl MetalProgressEvent {
    #[must_use]
    pub fn load_weights(current: u64, total: u64, detail: String) -> Self {
        Self {
            stage: MetalProgressStage::LoadWeights,
            current,
            total,
            unit: MetalProgressUnit::Byte,
            detail,
        }
    }

    #[must_use]
    pub fn prefill_tokens(current: usize, total: usize) -> Self {
        Self {
            stage: MetalProgressStage::PrefillTokens,
            current: current as u64,
            total: total as u64,
            unit: MetalProgressUnit::Token,
            detail: format!("token {current}/{total}"),
        }
    }
}
