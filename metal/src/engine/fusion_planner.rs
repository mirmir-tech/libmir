use super::{Stream, lowering::FeedForwardLowering};
use crate::{FusionMode, MetalFusionConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectionFusion {
    pub attention: bool,
    pub key_value: bool,
    pub gate_up: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct ProjectionBiases {
    attention: bool,
    key_value: bool,
    gate_up: bool,
}

impl ProjectionBiases {
    pub fn new(query_key: [bool; 2], value: Option<bool>, gate_up: [bool; 2]) -> Self {
        let [query, key] = query_key;
        let [gate, up] = gate_up;
        Self {
            attention: query || key || value.is_some_and(|biased| biased),
            key_value: key || value.is_none_or(|biased| biased),
            gate_up: gate || up,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatedDeltaFusion {
    pub compiled_normalized_decode: bool,
    pub recurrent_decode: bool,
    pub recurrent_normalization: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct FusionPlanner<'a> {
    config: &'a MetalFusionConfig,
}

impl<'a> FusionPlanner<'a> {
    pub fn new(stream: &'a Stream) -> Self {
        Self { config: &stream.config().fusion }
    }

    pub fn projections(
        self,
        feed_forward: FeedForwardLowering,
        biases: ProjectionBiases,
    ) -> ProjectionFusion {
        let (attention, gate_up, key_value) = match feed_forward {
            FeedForwardLowering::Dense => (
                self.config.dense_attention.enabled(),
                self.config.dense_gate_up.enabled(),
                false,
            ),
            FeedForwardLowering::DenseAndRouted => (
                self.config.hybrid_attention.enabled(),
                self.config.hybrid_dense_gate_up.enabled(),
                true,
            ),
            FeedForwardLowering::SharedRouted => (false, true, false),
            FeedForwardLowering::ClampedRouted => (false, false, false),
        };
        ProjectionFusion {
            attention: attention && !biases.attention,
            key_value: attention && key_value && !biases.key_value,
            gate_up: gate_up && !biases.gate_up,
        }
    }

    pub fn expert_mode(self, feed_forward: FeedForwardLowering) -> FusionMode {
        match feed_forward {
            FeedForwardLowering::DenseAndRouted => self.config.routed_expert_gate_up,
            FeedForwardLowering::SharedRouted => self.config.shared_expert_gate_up,
            FeedForwardLowering::Dense | FeedForwardLowering::ClampedRouted => FusionMode::Disabled,
        }
    }

    pub fn shared_dense_gate_up_mode(self) -> FusionMode {
        if self.config.shared_dense_gate_up.enabled() {
            FusionMode::Enabled
        } else {
            FusionMode::Auto
        }
    }

    pub fn native_router(self, feed_forward: FeedForwardLowering) -> bool {
        feed_forward == FeedForwardLowering::DenseAndRouted && self.config.native_router.enabled()
    }

    pub fn gated_delta(self) -> GatedDeltaFusion {
        let recurrent_decode = self.config.fused_gated_delta_decode.enabled();
        let recurrent_normalization = self.config.fused_gated_delta_normalization.enabled();
        GatedDeltaFusion {
            compiled_normalized_decode: self.config.compiled_gated_delta_decode.enabled()
                && recurrent_decode
                && recurrent_normalization,
            recurrent_decode,
            recurrent_normalization,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{FeatureToggle, MetalConfig};

    #[test]
    fn admits_fusions_by_operation_and_bias_contract() -> super::super::Result<()> {
        let stream = Stream::new_gpu_with_config(Arc::new(MetalConfig::default()))?;
        let planner = FusionPlanner::new(&stream);
        let unbiased = ProjectionBiases::new([false, false], Some(false), [false, false]);

        assert_eq!(
            planner.projections(FeedForwardLowering::DenseAndRouted, unbiased),
            ProjectionFusion {
                attention: true,
                key_value: true,
                gate_up: true
            }
        );
        assert_eq!(
            planner.projections(
                FeedForwardLowering::Dense,
                ProjectionBiases::new([true, false], Some(false), [false, false])
            ),
            ProjectionFusion {
                attention: false,
                key_value: false,
                gate_up: true
            }
        );
        assert_eq!(
            planner.projections(
                FeedForwardLowering::DenseAndRouted,
                ProjectionBiases::new([false, false], None, [false, false])
            ),
            ProjectionFusion {
                attention: true,
                key_value: false,
                gate_up: true
            }
        );
        assert_eq!(planner.expert_mode(FeedForwardLowering::Dense), FusionMode::Disabled);
        assert!(planner.native_router(FeedForwardLowering::DenseAndRouted));
        assert!(!planner.native_router(FeedForwardLowering::SharedRouted));
        Ok(())
    }

    #[test]
    fn configuration_can_disable_an_admissible_operation() -> super::super::Result<()> {
        let mut config = MetalConfig::default();
        config.fusion.dense_gate_up = FeatureToggle::Disabled;
        let stream = Stream::new_gpu_with_config(Arc::new(config))?;
        let fusion = FusionPlanner::new(&stream).projections(
            FeedForwardLowering::Dense,
            ProjectionBiases::new([false, false], Some(false), [false, false]),
        );

        assert!(!fusion.gate_up);
        Ok(())
    }

    #[test]
    fn compiled_gated_delta_requires_the_complete_fused_path() -> super::super::Result<()> {
        let mut config = MetalConfig::default();
        config.fusion.fused_gated_delta_normalization = FeatureToggle::Disabled;
        let stream = Stream::new_gpu_with_config(Arc::new(config))?;
        let fusion = FusionPlanner::new(&stream).gated_delta();

        assert_eq!(
            fusion,
            GatedDeltaFusion {
                compiled_normalized_decode: false,
                recurrent_decode: true,
                recurrent_normalization: false,
            }
        );
        Ok(())
    }
}
