//! Role parsing for checkpoints whose blocks hold either a mixer or a
//! feed-forward, such as state-space hybrid decoders.

use super::{LayerTensorRole, LogicalTensorRole, parse::layer};
use crate::semantic::{DecoderSpec, LayerBlockPosition};

/// Parses a path below the `backbone.` namespace.
pub(super) fn parse(path: &str, decoder: &DecoderSpec) -> LogicalTensorRole {
    match path {
        "embeddings.weight" => return LogicalTensorRole::Embedding,
        "norm_f.weight" => return LogicalTensorRole::FinalNorm,
        _ => {},
    }
    let auxiliary = || LogicalTensorRole::Auxiliary { path: format!("backbone.{path}") };
    let Some((block, suffix)) = path
        .strip_prefix("layers.")
        .and_then(|remainder| remainder.split_once('.'))
        .and_then(|(block, suffix)| Some((block.parse::<usize>().ok()?, suffix)))
    else {
        return auxiliary();
    };
    let Some((index, position)) = decoder.block_layout.locate(&decoder.layers, block) else {
        return auxiliary();
    };
    let tensor = match (suffix, position) {
        ("norm.weight", LayerBlockPosition::Mixer) => LayerTensorRole::InputNorm,
        ("norm.weight", LayerBlockPosition::FeedForward) => LayerTensorRole::PostAttentionNorm,
        (suffix, position) => {
            let tensor = layer(suffix);
            // A tensor in the wrong kind of block gets no layer role; the
            // binding grammar rejects it instead of misusing it.
            if feed_forward_role(&tensor) != (position == LayerBlockPosition::FeedForward) {
                return auxiliary();
            }
            tensor
        },
    };
    LogicalTensorRole::Layer { index, tensor }
}

const fn feed_forward_role(tensor: &LayerTensorRole) -> bool {
    matches!(
        tensor,
        LayerTensorRole::Router
            | LayerTensorRole::RouterCorrectionBias
            | LayerTensorRole::LatentProjection { .. }
            | LayerTensorRole::ExpertProjection { .. }
            | LayerTensorRole::SharedExpertProjection { .. }
    )
}
