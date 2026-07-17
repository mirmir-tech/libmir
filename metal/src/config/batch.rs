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

#[derive(Debug, Clone, Default)]
pub struct MetalBatchConfig {
    pub dense_decode: DenseBatchMode,
}
