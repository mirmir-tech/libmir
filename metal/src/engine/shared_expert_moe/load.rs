use models::{
    layout::DecoderConfig,
    weights::{SharedRoutedFeedForwardBindings, TensorBinding},
};

use super::{ExpertRouter, RoutedGateUp, ScoredRouter, SharedExpertMoe, SharedExpertMoeConfig};
use crate::{
    FusionMode,
    engine::{
        Dtype, Error, ModelTensors, QuantizedLinear, Result, Stream,
        binding::BoundLinear,
        fusion_planner::{FusionPlanner, ProjectionBiases},
        lowering::FeedForwardLowering,
    },
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
        let shared_gate = BoundLinear::load(tensors, bindings.shared_gate, stream)?;
        let shared_up = BoundLinear::load(tensors, bindings.shared_up, stream)?;
        let planner = FusionPlanner::new(stream);
        let fusion = planner.projections(lowering, projection_biases(&shared_gate, &shared_up));
        Ok(Self {
            config,
            router: BoundLinear::load(tensors, bindings.router, stream)?,
            routed_gate_up,
            routed_down,
            shared_gate,
            shared_up,
            fused_shared_gate_up: None,
            shared_down: BoundLinear::load(tensors, bindings.shared_down, stream)?,
            shared_output_gate: bindings
                .shared_output_gate
                .map(|gate| BoundLinear::load(tensors, gate, stream))
                .transpose()?,
            shared_gate_up_mode: if fusion.gate_up {
                planner.shared_dense_gate_up_mode()
            } else {
                FusionMode::Disabled
            },
            routing,
        })
    }

    /// The decoder's experts and router: the `DeepSeek` layout's configured
    /// router, else renormalized softmax top-k.
    pub fn load_decoder(
        tensors: &ModelTensors,
        decoder: &DecoderConfig,
        bindings: SharedRoutedFeedForwardBindings<'_>,
        lowering: FeedForwardLowering,
        stream: &Stream,
    ) -> Result<Self> {
        let missing = |what: &str| Error::InvalidModel(format!("missing MoE {what}"));
        let config = SharedExpertMoeConfig::new(
            decoder.num_experts.ok_or_else(|| missing("expert count"))?,
            decoder.top_k_experts.ok_or_else(|| missing("top-k"))?,
            decoder.moe_intermediate_size.unwrap_or(decoder.intermediate_size),
        )?;
        let routing = router(tensors, decoder, bindings.router_correction_bias, stream)?;
        Self::load_bindings(tensors, bindings, config, lowering, routing, stream)
    }
}

fn router(
    tensors: &ModelTensors,
    decoder: &DecoderConfig,
    correction_bias: Option<&TensorBinding>,
    stream: &Stream,
) -> Result<ExpertRouter> {
    let Some(experts) = decoder.routed_experts.as_ref() else {
        return Ok(match decoder.expert_router {
            models::layout::ExpertRouter::SigmoidInputScaled => {
                if decoder.top_k_experts != Some(1) {
                    return Err(Error::InvalidModel(
                        "input-scaled routing selects one expert per token".into(),
                    ));
                }
                ExpertRouter::SigmoidInputScaled
            },
            models::layout::ExpertRouter::SoftmaxTopK => ExpertRouter::Unit,
            models::layout::ExpertRouter::SparseMixer { .. } => {
                return Err(Error::InvalidModel(
                    "shared-expert layers have no sparsemixer router".into(),
                ));
            },
        });
    };
    Ok(ExpertRouter::Scored(ScoredRouter {
        routing: experts.routing,
        correction_bias: correction_bias
            .map(|bias| tensors.get(&bias.source)?.astype(Dtype::Float32, stream))
            .transpose()?,
        group_limit: experts.group_limit,
        output_scale: experts.output_scale.to_string().parse()?,
    }))
}
