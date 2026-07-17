#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FusionMode {
    #[default]
    Auto,
    Enabled,
    Disabled,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FeatureToggle {
    Enabled,
    #[default]
    Disabled,
}

impl FeatureToggle {
    #[must_use]
    pub const fn enabled(self) -> bool {
        matches!(self, Self::Enabled)
    }
}

impl From<bool> for FeatureToggle {
    fn from(value: bool) -> Self {
        if value {
            Self::Enabled
        } else {
            Self::Disabled
        }
    }
}

#[derive(Debug, Clone)]
pub struct MetalFusionConfig {
    pub device_token_pipeline: FeatureToggle,
    pub hybrid_attention: FeatureToggle,
    pub hybrid_dense_gate_up: FeatureToggle,
    pub dense_attention: FeatureToggle,
    pub dense_gate_up: FeatureToggle,
    pub routed_expert_gate_up: FusionMode,
    pub shared_expert_gate_up: FusionMode,
    pub shared_dense_gate_up: FeatureToggle,
    pub native_router: FeatureToggle,
    pub compiled_gated_delta_decode: FeatureToggle,
    pub fused_gated_delta_decode: FeatureToggle,
    pub fused_gated_delta_normalization: FeatureToggle,
}

impl Default for MetalFusionConfig {
    fn default() -> Self {
        Self {
            device_token_pipeline: FeatureToggle::Enabled,
            hybrid_attention: FeatureToggle::Enabled,
            hybrid_dense_gate_up: FeatureToggle::Enabled,
            dense_attention: FeatureToggle::Enabled,
            dense_gate_up: FeatureToggle::Enabled,
            routed_expert_gate_up: FusionMode::Auto,
            shared_expert_gate_up: FusionMode::Auto,
            shared_dense_gate_up: FeatureToggle::Disabled,
            native_router: FeatureToggle::Enabled,
            compiled_gated_delta_decode: FeatureToggle::Enabled,
            fused_gated_delta_decode: FeatureToggle::Enabled,
            fused_gated_delta_normalization: FeatureToggle::Enabled,
        }
    }
}
