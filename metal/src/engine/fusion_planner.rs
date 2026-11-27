use super::{
    Stream,
    lowering::{Activation, FeedForwardLowering},
};
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

    /// Fusions of a layer whose feed-forward block is `feed_forward`, in the
    /// dense, mixed, shared-routed and clamped-routed runtimes.
    pub fn projections(
        self,
        feed_forward: &FeedForwardLowering,
        biases: ProjectionBiases,
    ) -> ProjectionFusion {
        let (attention, gate_up) = match feed_forward {
            FeedForwardLowering::Dense {
                activation: Activation::SwiGlu { clamp: None } | Activation::GeluTanh,
            } => (self.config.dense_attention.enabled(), self.config.dense_gate_up.enabled()),
            FeedForwardLowering::Routed { experts, shared: None }
                if experts.activation == (Activation::SwiGlu { clamp: None }) =>
            {
                (self.config.dense_attention.enabled(), false)
            },
            FeedForwardLowering::Routed { shared: Some(_), .. } => (false, true),
            FeedForwardLowering::Dense { .. }
            | FeedForwardLowering::Routed { .. }
            | FeedForwardLowering::DenseAndRouted { .. }
            | FeedForwardLowering::LatentRouted { .. }
            | FeedForwardLowering::None => (false, false),
        };
        ProjectionFusion {
            attention: attention && !biases.attention,
            key_value: false,
            gate_up: gate_up && !biases.gate_up,
        }
    }

    /// Fusions of a layer in the hybrid runtimes (`DenseAndRouted` and
    /// `DenseGelu`), whose attention also fuses keys with values.
    pub fn hybrid_projections(self, biases: ProjectionBiases) -> ProjectionFusion {
        let attention = self.config.hybrid_attention.enabled();
        ProjectionFusion {
            attention: attention && !biases.attention,
            key_value: attention && !biases.key_value,
            gate_up: self.config.hybrid_dense_gate_up.enabled() && !biases.gate_up,
        }
    }

    /// How the gate and up projections of the experts beside the hybrid
    /// runtimes' dense blocks fuse.
    pub const fn hybrid_expert_mode(self) -> FusionMode {
        self.config.routed_expert_gate_up
    }

    /// How a shared expert's gate and up projections fuse; routed blocks
    /// without one fuse none.
    pub const fn expert_mode(self, feed_forward: &FeedForwardLowering) -> FusionMode {
        match feed_forward {
            FeedForwardLowering::Routed { shared: Some(_), .. } => {
                self.config.shared_expert_gate_up
            },
            FeedForwardLowering::Dense { .. }
            | FeedForwardLowering::Routed { shared: None, .. }
            | FeedForwardLowering::DenseAndRouted { .. }
            | FeedForwardLowering::LatentRouted { .. }
            | FeedForwardLowering::None => FusionMode::Disabled,
        }
    }

    /// How a shared expert's gate and up projections fuse at decode.
    pub fn shared_expert_gate_up_mode(self, biases: ProjectionBiases) -> FusionMode {
        if biases.gate_up {
            FusionMode::Disabled
        } else {
            self.shared_dense_gate_up_mode()
        }
    }

    pub fn shared_dense_gate_up_mode(self) -> FusionMode {
        if self.config.shared_dense_gate_up.enabled() {
            FusionMode::Enabled
        } else {
            FusionMode::Auto
        }
    }

    /// Whether the dense-and-routed runtime routes with the native kernel.
    pub fn native_router(self) -> bool {
        self.config.native_router.enabled()
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

    const DENSE: FeedForwardLowering = FeedForwardLowering::Dense {
        activation: Activation::SwiGlu { clamp: None },
    };

    #[test]
    fn admits_fusions_by_operation_and_bias_contract() -> super::super::Result<()> {
        let stream = Stream::new_gpu_with_config(Arc::new(MetalConfig::default()))?;
        let planner = FusionPlanner::new(&stream);
        let unbiased = ProjectionBiases::new([false, false], Some(false), [false, false]);

        assert_eq!(
            planner.hybrid_projections(unbiased),
            ProjectionFusion {
                attention: true,
                key_value: true,
                gate_up: true
            }
        );
        assert_eq!(
            planner.projections(
                &DENSE,
                ProjectionBiases::new([true, false], Some(false), [false, false])
            ),
            ProjectionFusion {
                attention: false,
                key_value: false,
                gate_up: true
            }
        );
        assert_eq!(
            planner.hybrid_projections(ProjectionBiases::new([false, false], None, [false, false])),
            ProjectionFusion {
                attention: true,
                key_value: false,
                gate_up: true
            }
        );
        assert_eq!(
            planner.projections(
                &FeedForwardLowering::Dense { activation: Activation::ReluSquared },
                unbiased
            ),
            ProjectionFusion {
                attention: false,
                key_value: false,
                gate_up: false
            }
        );
        assert_eq!(planner.expert_mode(&DENSE), FusionMode::Disabled);
        assert!(planner.native_router());
        Ok(())
    }

    #[test]
    fn configuration_can_disable_an_admissible_operation() -> super::super::Result<()> {
        let mut config = MetalConfig::default();
        config.fusion.dense_gate_up = FeatureToggle::Disabled;
        let stream = Stream::new_gpu_with_config(Arc::new(config))?;
        let fusion = FusionPlanner::new(&stream).projections(
            &DENSE,
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
