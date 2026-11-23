//! Decoder features only the dense graph decoder executes; the other CUDA
//! runtimes refuse them rather than run a different model.

use models::semantic::{
    AttentionWindow, FeedForwardSpec, MixerSpec, NormalizationKind, QkNormalization,
    RouterNormalization, SemanticModelSpec,
};

use super::CudaDecoderRuntime;
use crate::{ArchitectureError, Result};

pub(super) fn require_graph_decoder(
    semantic: &SemanticModelSpec,
    runtime: CudaDecoderRuntime,
) -> Result<()> {
    let layers = &semantic.decoder.layers;
    if runtime == CudaDecoderRuntime::Dense {
        return Ok(());
    }
    let attention = || {
        layers.iter().filter_map(|layer| match &layer.mixer {
            MixerSpec::SoftmaxAttention(attention) => Some(attention),
            _ => None,
        })
    };
    #[allow(clippy::float_cmp)]
    let refusal = if semantic.decoder.residual_multiplier != 1.0
        || semantic.decoder.logit_divisor != 1.0
    {
        Some("folds residual multipliers and logit divisors")
    } else if attention().any(|attention| attention.logit_softcap.is_some()) {
        Some("caps attention scores")
    } else if layers.iter().any(|layer| !layer.norm_placement.is_pre()) {
        Some("runs post-norm and parallel blocks")
    } else if semantic.decoder.final_norm.kind == NormalizationKind::Layer
        || layers.iter().any(|layer| layer.input_norm.kind == NormalizationKind::Layer)
    {
        Some("runs LayerNorm decoders")
    } else if attention()
        .any(|attention| attention.qk_normalization == QkNormalization::QueryKeyRmsAcrossHeads)
    {
        Some("normalizes queries and keys across heads")
    } else if layers.iter().any(|layer| {
        matches!(
            &layer.feed_forward,
            FeedForwardSpec::Routed { routed, .. }
                if matches!(routed.router_normalization, RouterNormalization::SparseMixer { .. })
        )
    }) {
        Some("routes experts by sparsemixer")
    } else if attention().any(|attention| {
        matches!(attention.window, Some(AttentionWindow::Chunked { .. }))
            || attention.query_temperature.is_some()
            || attention.qk_normalization == QkNormalization::QueryKeyUnitRms
    }) {
        Some("runs chunked attention, query temperature and unit Q/K norms")
    } else if layers.iter().any(|layer| {
        matches!(
            &layer.feed_forward,
            FeedForwardSpec::Routed { routed, .. }
                if routed.router_normalization == RouterNormalization::SigmoidInputScaledTopK
        )
    }) {
        Some("scales expert inputs by their router")
    } else {
        None
    };
    refusal.map_or(Ok(()), |feature| {
        Err(ArchitectureError::invalid(format!(
            "CUDA {feature} only in the dense graph decoder"
        )))
    })
}
