//! The `DeepSeek` expert layout in semantic form (`DeepSeek`, Kimi, GLM-4.5).

use super::{activation, invalid};
use crate::{
    error::Result,
    layout::{DecoderConfig, ExpertRouting},
    semantic::{FeedForwardSpec, RoutedExpertsSpec, RouterNormalization, SharedExpertSpec},
};

/// The layer's feed-forward: dense before `first_dense_layers` or without
/// routed experts, else routed experts plus shared ones.
pub(super) fn feed_forward(decoder: &DecoderConfig, index: usize) -> Result<FeedForwardSpec> {
    let activation = activation(decoder);
    let experts = decoder.routed_experts.as_ref();
    let Some(experts) = experts.filter(|experts| index >= experts.first_dense_layers) else {
        return Ok(FeedForwardSpec::Dense {
            intermediate_size: decoder.intermediate_size,
            activation,
        });
    };
    let expert_count = decoder
        .num_experts
        .ok_or_else(|| invalid("routed decoder is missing its expert count"))?;
    let top_k = decoder
        .top_k_experts
        .ok_or_else(|| invalid("routed decoder is missing its top-k expert count"))?;
    let intermediate_size = decoder
        .moe_intermediate_size
        .ok_or_else(|| invalid("routed decoder is missing moe_intermediate_size"))?;
    let shared = experts
        .shared_experts
        .map(|count| {
            Ok::<_, crate::error::ModelsError>(SharedExpertSpec {
                intermediate_size: count
                    .get()
                    .checked_mul(intermediate_size)
                    .ok_or_else(|| invalid("shared expert width overflows"))?,
                activation: activation.clone(),
                gated_output: false,
            })
        })
        .transpose()?;
    Ok(FeedForwardSpec::Routed {
        routed: RoutedExpertsSpec {
            expert_count,
            top_k,
            intermediate_size,
            activation,
            router_normalization: match experts.routing {
                ExpertRouting::Softmax { renormalize: true } => RouterNormalization::SoftmaxTopK,
                ExpertRouting::Softmax { renormalize: false } => {
                    RouterNormalization::SoftmaxProbabilityTopK
                },
                ExpertRouting::SigmoidWithCorrectionBias => RouterNormalization::SigmoidBiasedTopK,
            },
            group_limit: experts.group_limit,
            output_scale: experts.output_scale,
        },
        shared,
    })
}
