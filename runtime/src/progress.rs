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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgressCount {
    current: u64,
    total: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgressEvent {
    LoadWeights { count: ProgressCount, detail: String },
    InitializeRuntime { count: ProgressCount, detail: String },
    Warmup { count: ProgressCount, detail: String },
    PrefillTokens { count: ProgressCount, detail: String },
    DecodeTokens { count: ProgressCount, detail: String },
}

impl ProgressCount {
    #[must_use]
    pub const fn new(current: u64, total: u64) -> Self {
        Self { current, total }
    }

    #[must_use]
    pub const fn current(self) -> u64 {
        self.current
    }

    #[must_use]
    pub const fn total(self) -> u64 {
        self.total
    }
}

impl ProgressEvent {
    #[must_use]
    pub fn load_weights(current: u64, total: u64, detail: impl Into<String>) -> Self {
        Self::LoadWeights {
            count: ProgressCount::new(current, total),
            detail: detail.into(),
        }
    }

    #[must_use]
    pub fn initialize_runtime(current: u64, total: u64, detail: impl Into<String>) -> Self {
        Self::InitializeRuntime {
            count: ProgressCount::new(current, total),
            detail: detail.into(),
        }
    }

    #[must_use]
    pub fn warmup(current: usize, total: usize, detail: impl Into<String>) -> Self {
        Self::Warmup {
            count: ProgressCount::new(current as u64, total as u64),
            detail: detail.into(),
        }
    }

    #[must_use]
    pub fn prefill_tokens(current: usize, total: usize) -> Self {
        Self::PrefillTokens {
            count: ProgressCount::new(current as u64, total as u64),
            detail: format!("token {current}/{total}"),
        }
    }

    #[must_use]
    pub fn decode_tokens(current: usize, total: usize) -> Self {
        Self::DecodeTokens {
            count: ProgressCount::new(current as u64, total as u64),
            detail: format!("token {current}/{total}"),
        }
    }

    #[must_use]
    pub const fn stage(&self) -> ProgressStage {
        match self {
            Self::LoadWeights { .. } => ProgressStage::LoadWeights,
            Self::InitializeRuntime { .. } => ProgressStage::InitializeRuntime,
            Self::Warmup { .. } => ProgressStage::Warmup,
            Self::PrefillTokens { .. } => ProgressStage::PrefillTokens,
            Self::DecodeTokens { .. } => ProgressStage::DecodeTokens,
        }
    }

    #[must_use]
    pub const fn unit(&self) -> ProgressUnit {
        match self {
            Self::LoadWeights { .. } => ProgressUnit::Byte,
            Self::InitializeRuntime { .. } | Self::Warmup { .. } => ProgressUnit::Item,
            Self::PrefillTokens { .. } | Self::DecodeTokens { .. } => ProgressUnit::Token,
        }
    }

    #[must_use]
    pub const fn count(&self) -> ProgressCount {
        match self {
            Self::LoadWeights { count, .. }
            | Self::InitializeRuntime { count, .. }
            | Self::Warmup { count, .. }
            | Self::PrefillTokens { count, .. }
            | Self::DecodeTokens { count, .. } => *count,
        }
    }

    #[must_use]
    pub fn detail(&self) -> &str {
        match self {
            Self::LoadWeights { detail, .. }
            | Self::InitializeRuntime { detail, .. }
            | Self::Warmup { detail, .. }
            | Self::PrefillTokens { detail, .. }
            | Self::DecodeTokens { detail, .. } => detail,
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
