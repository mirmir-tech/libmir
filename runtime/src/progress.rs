use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressStage {
    LoadWeights,
    InitializeRuntime,
    Warmup,
    PrefillTokens,
    DecodeTokens,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressUnit {
    Item,
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
    pub fn initialize_runtime(current: u64, total: u64, detail: impl Into<String>) -> Self {
        Self {
            stage: ProgressStage::InitializeRuntime,
            current,
            total,
            unit: ProgressUnit::Item,
            detail: detail.into(),
        }
    }

    #[must_use]
    pub fn warmup(current: usize, total: usize, detail: impl Into<String>) -> Self {
        Self {
            stage: ProgressStage::Warmup,
            current: current as u64,
            total: total as u64,
            unit: ProgressUnit::Item,
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

impl ProgressStage {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LoadWeights => "loading",
            Self::InitializeRuntime => "initializing",
            Self::Warmup => "warming",
            Self::PrefillTokens => "prefill",
            Self::DecodeTokens => "decode",
        }
    }
}

impl fmt::Display for ProgressStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl ProgressUnit {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Item => "item",
            Self::Byte => "byte",
            Self::Token => "token",
        }
    }
}

impl fmt::Display for ProgressUnit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}
