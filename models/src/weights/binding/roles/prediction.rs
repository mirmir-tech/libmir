//! Role parsing for multi-token prediction modules (`mtp.layers.N.*`).
//! Module `d` spans block `2d`, the attention mixer with the fusion inputs,
//! and block `2d + 1`, the routed block with the output norm.

use super::{AttentionProjectionRole, LogicalTensorRole, PredictionTensorRole, paired::block_role};
use crate::semantic::{DecoderSpec, LayerBlockPosition};

/// Parses a path below the `mtp.` namespace.
pub(super) fn parse(path: &str, decoder: &DecoderSpec) -> LogicalTensorRole {
    let auxiliary = || LogicalTensorRole::Auxiliary { path: format!("mtp.{path}") };
    if !matches!(decoder.block_layout, crate::semantic::BlockLayoutSpec::Paired { .. }) {
        return single(path, decoder).unwrap_or_else(auxiliary);
    }
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

/// Qwen3.5's `mtp`: the fusion inputs and output norm at the namespace root
/// of its one module, and `layers.N` holding ordinary decoder layers.
fn single(path: &str, decoder: &DecoderSpec) -> Option<LogicalTensorRole> {
    let root = match path {
        "fc.weight" => Some(PredictionTensorRole::Fusion),
        "pre_fc_norm_embedding.weight" => Some(PredictionTensorRole::EmbeddingNorm),
        "pre_fc_norm_hidden.weight" => Some(PredictionTensorRole::HiddenNorm),
        "norm.weight" => Some(PredictionTensorRole::FinalNorm),
        _ => None,
    };
    if let Some(tensor) = root {
        return (!decoder.prediction_layers.is_empty())
            .then_some(LogicalTensorRole::Prediction { depth: 0, tensor });
    }
    let (depth, suffix) = path.strip_prefix("layers.")?.split_once('.')?;
    let depth = depth.parse::<usize>().ok()?;
    (depth < decoder.prediction_layers.len()).then(|| LogicalTensorRole::Prediction {
        depth,
        tensor: PredictionTensorRole::Layer { tensor: super::parse::layer(suffix) },
    })
}
