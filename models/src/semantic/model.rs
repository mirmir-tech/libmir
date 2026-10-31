use serde::{Deserialize, Serialize};

use super::{FeedForwardSpec, MixerSpec};

pub const CURRENT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticModelSpec {
    pub schema_version: u32,
    pub decoder: DecoderSpec,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderSpec {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub tie_word_embeddings: bool,
    pub final_norm: NormalizationSpec,
    pub layers: Vec<DecoderLayerSpec>,
    #[serde(default, skip_serializing_if = "BlockLayoutSpec::is_unified")]
    pub block_layout: BlockLayoutSpec,
    /// Multi-token prediction modules after the decoder, each an attention
    /// mixer with the routed block after it; empty when the model has none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prediction_layers: Vec<DecoderLayerSpec>,
}

/// How checkpoint blocks map to semantic decoder layers.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BlockLayoutSpec {
    /// Checkpoint layer `i` is semantic layer `i`.
    #[default]
    Unified,
    /// Each semantic layer is composed of its mixer block and, when the layer
    /// has a feed-forward, the block directly after it.
    Paired {
        physical_blocks: usize,
        mixer_blocks: Vec<usize>,
    },
}

/// Position of a checkpoint block inside its semantic layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerBlockPosition {
    Mixer,
    FeedForward,
}

impl BlockLayoutSpec {
    #[must_use]
    pub const fn is_unified(&self) -> bool {
        matches!(self, Self::Unified)
    }

    /// Maps a paired checkpoint block to its semantic layer and position.
    /// Returns `None` for a unified layout or a block outside the pairing.
    #[must_use]
    pub fn locate(
        &self,
        layers: &[DecoderLayerSpec],
        block: usize,
    ) -> Option<(usize, LayerBlockPosition)> {
        let Self::Paired { mixer_blocks, .. } = self else {
            return None;
        };
        mixer_blocks.iter().zip(layers).enumerate().find_map(|(index, (mixer, layer))| {
            if *mixer == block {
                Some((index, LayerBlockPosition::Mixer))
            } else if *mixer + 1 == block && layer.feed_forward != FeedForwardSpec::None {
                Some((index, LayerBlockPosition::FeedForward))
            } else {
                None
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderLayerSpec {
    pub index: usize,
    pub input_norm: NormalizationSpec,
    pub post_attention_norm: NormalizationSpec,
    pub mixer: MixerSpec,
    pub feed_forward: FeedForwardSpec,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NormalizationSpec {
    pub kind: NormalizationKind,
    pub epsilon: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationKind {
    Rms,
    Layer,
}
