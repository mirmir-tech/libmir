use models::{
    semantic::{RoutedExpertsSpec, RouterNormalization},
    weights::{RoutedFeedForwardBindings, SharedRoutedFeedForwardBindings, TensorBinding},
};

use super::{
    ExpertRouter, RoutedGateUp, ScoredRouter, ScoredRouting, SharedExpert, SharedExpertMoe,
    SharedExpertMoeConfig,
};
use crate::engine::{
    Dtype, ModelTensors, QuantizedLinear, Result, Stream, binding::BoundLinear,
    fusion_planner::ProjectionBiases, lowering::FeedForwardLowering,
};

pub(super) fn projection_biases(gate: &BoundLinear, up: &BoundLinear) -> ProjectionBiases {
    ProjectionBiases::new([false, false], None, [gate.has_bias(), up.has_bias()])
}

pub(super) fn linear(
    tensors: &ModelTensors,
    prefix: &str,
    name: &str,
    group_size: i32,
) -> Result<BoundLinear> {
    QuantizedLinear::load(tensors, &format!("{prefix}.{name}"), group_size).map(BoundLinear::Affine)
}

impl SharedExpertMoe {
    pub fn load_bindings(
        tensors: &ModelTensors,
        bindings: SharedRoutedFeedForwardBindings<'_>,
        config: SharedExpertMoeConfig,
        lowering: FeedForwardLowering,
        routing: ExpertRouter,
        stream: &Stream,
    ) -> Result<Self> {
        let (routed_gate_up, routed_down) = RoutedGateUp::load(tensors, bindings.routed, stream)?;
        let shared = SharedExpert::new(
            [
                BoundLinear::load(tensors, bindings.shared_gate, stream)?,
                BoundLinear::load(tensors, bindings.shared_up, stream)?,
                BoundLinear::load(tensors, bindings.shared_down, stream)?,
            ],
            bindings
                .shared_output_gate
                .map(|gate| BoundLinear::load(tensors, gate, stream))
                .transpose()?,
            lowering,
            stream,
        );
        Ok(Self {
            config,
            router: BoundLinear::load(tensors, bindings.router, stream)?,
            routed_gate_up,
            routed_down,
            shared: Some(shared),
            routing,
        })
    }

    /// The layer's routed and shared experts and router from its
    /// specification.
    pub fn load_spec(
        tensors: &ModelTensors,
        routed: &RoutedExpertsSpec,
        bindings: SharedRoutedFeedForwardBindings<'_>,
        lowering: FeedForwardLowering,
        stream: &Stream,
    ) -> Result<Self> {
        let routing = router(tensors, routed, bindings.router_correction_bias, stream)?;
        Self::load_bindings(tensors, bindings, config(routed)?, lowering, routing, stream)
    }

    /// Routed experts alone, with no shared expert (Qwen3-MoE, Mixtral,
    /// Phi-3.5-MoE), from the layer's specification.
    pub fn load_routed_spec(
        tensors: &ModelTensors,
        routed: &RoutedExpertsSpec,
        bindings: RoutedFeedForwardBindings<'_>,
        stream: &Stream,
    ) -> Result<Self> {
        let (routed_gate_up, routed_down) = RoutedGateUp::load(tensors, bindings.experts, stream)?;
        Ok(Self {
            config: config(routed)?,
            router: BoundLinear::load(tensors, bindings.router, stream)?,
            routed_gate_up,
            routed_down,
            shared: None,
            routing: router(tensors, routed, None, stream)?,
        })
    }
}

fn config(routed: &RoutedExpertsSpec) -> Result<SharedExpertMoeConfig> {
    SharedExpertMoeConfig::new(routed.expert_count, routed.top_k, routed.intermediate_size)
}

/// The router a specification describes. The semantics already require one
/// expert for input-scaled routing and a correction bias for the biased
/// sigmoid.
fn router(
    tensors: &ModelTensors,
    routed: &RoutedExpertsSpec,
    correction_bias: Option<&TensorBinding>,
    stream: &Stream,
) -> Result<ExpertRouter> {
    let scored = |routing| -> Result<ExpertRouter> {
        Ok(ExpertRouter::Scored(ScoredRouter {
            routing,
            correction_bias: correction_bias
                .map(|bias| tensors.get(&bias.source)?.astype(Dtype::Float32, stream))
                .transpose()?,
            group_limit: routed.group_limit,
            output_scale: routed.output_scale.to_string().parse()?,
        }))
    };
    match routed.router_normalization {
        RouterNormalization::SoftmaxTopK
            if routed.group_limit.is_none()
                && routed.output_scale.to_bits() == 1.0_f64.to_bits() =>
        {
            Ok(ExpertRouter::Unit)
        },
        RouterNormalization::SoftmaxTopK => scored(ScoredRouting::Softmax { renormalize: true }),
        RouterNormalization::SoftmaxProbabilityTopK => {
            scored(ScoredRouting::Softmax { renormalize: false })
        },
        RouterNormalization::SigmoidBiasedTopK => scored(ScoredRouting::SigmoidWithCorrectionBias),
        RouterNormalization::SigmoidInputScaledTopK => Ok(ExpertRouter::SigmoidInputScaled),
        RouterNormalization::SparseMixer { jitter_eps } => {
            Ok(ExpertRouter::SparseMixer { jitter: jitter_eps.to_string().parse()? })
        },
    }
}
