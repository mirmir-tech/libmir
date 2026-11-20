#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DenseBatchMode {
    #[default]
    Auto,
    Packed,
    Scalar,
}

impl DenseBatchMode {
    pub(crate) const fn packed(self) -> bool {
        matches!(self, Self::Auto | Self::Packed)
    }
}

/// How dense decode stores K/V history for common-position rows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DenseHistoryMode {
    /// Appends one `[B, H, T, D]` tensor per layer and runs one SDPA call.
    #[default]
    Persistent,
    /// Keeps per-row paged attention for every step.
    PerRow,
}

impl DenseHistoryMode {
    pub(crate) const fn persistent(self) -> bool {
        matches!(self, Self::Persistent)
    }
}

#[derive(Debug, Clone, Default)]
pub struct MetalBatchConfig {
    pub dense_decode: DenseBatchMode,
    pub dense_history: DenseHistoryMode,
}
