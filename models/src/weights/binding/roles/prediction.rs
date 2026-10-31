//! Role parsing for multi-token prediction modules (`mtp.layers.N.*`).
//! Module `d` spans block `2d`, the attention mixer with the fusion inputs,
//! and block `2d + 1`, the routed block with the output norm.

use super::{AttentionProjectionRole, LogicalTensorRole, PredictionTensorRole, paired::block_role};
use crate::semantic::{DecoderSpec, LayerBlockPosition};

/// Parses a path below the `mtp.` namespace.
pub(super) fn parse(path: &str, decoder: &DecoderSpec) -> LogicalTensorRole {
    let auxiliary = || LogicalTensorRole::Auxiliary { path: format!("mtp.{path}") };
    let Some((block, suffix)) = path
        .strip_prefix("layers.")
        .and_then(|remainder| remainder.split_once('.'))
        .and_then(|(block, suffix)| Some((block.parse::<usize>().ok()?, suffix)))
    else {
        return auxiliary();
    };
    let depth = block / 2;
    if depth >= decoder.prediction_layers.len() {
        return auxiliary();
    }
    let position = if block % 2 == 0 {
        LayerBlockPosition::Mixer
    } else {
        LayerBlockPosition::FeedForward
    };
    let tensor = match (suffix, position) {
        ("eh_proj.weight", LayerBlockPosition::Mixer) => PredictionTensorRole::Fusion,
        ("enorm.weight", LayerBlockPosition::Mixer) => PredictionTensorRole::EmbeddingNorm,
        ("hnorm.weight", LayerBlockPosition::Mixer) => PredictionTensorRole::HiddenNorm,
        ("mixer.k_proj.k_scale", LayerBlockPosition::Mixer) => {
            PredictionTensorRole::CacheScale { projection: AttentionProjectionRole::Key }
        },
        ("mixer.v_proj.v_scale", LayerBlockPosition::Mixer) => PredictionTensorRole::CacheScale {
            projection: AttentionProjectionRole::Value,
        },
        ("final_layernorm.weight", LayerBlockPosition::FeedForward) => {
            PredictionTensorRole::FinalNorm
        },
        (suffix, position) => match block_role(suffix, position) {
            Some(tensor) => PredictionTensorRole::Layer { tensor },
            None => return auxiliary(),
        },
    };
    LogicalTensorRole::Prediction { depth, tensor }
}
