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
    /// Factor applied to token embeddings before the first layer.
    #[serde(default = "unit_multiplier", skip_serializing_if = "is_unit_multiplier")]
    pub embedding_multiplier: f64,
    /// Factor on each residual branch before it joins the residual stream.
    #[serde(default = "unit_multiplier", skip_serializing_if = "is_unit_multiplier")]
    pub residual_multiplier: f64,
    /// Divisor of the output logits.
    #[serde(default = "unit_multiplier", skip_serializing_if = "is_unit_multiplier")]
    pub logit_divisor: f64,
    pub layers: Vec<DecoderLayerSpec>,
    #[serde(default, skip_serializing_if = "BlockLayoutSpec::is_unified")]
    pub block_layout: BlockLayoutSpec,
    /// Multi-token prediction modules after the decoder, each an attention
    /// mixer with the routed block after it; empty when the model has none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prediction_layers: Vec<DecoderLayerSpec>,
}

const fn unit_multiplier() -> f64 {
    1.0
}

#[allow(clippy::trivially_copy_pass_by_ref, clippy::float_cmp)]
const fn is_unit_multiplier(value: &f64) -> bool {
    *value == 1.0
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
    /// Where the block normalizes: before each sub-block, or only on each
    /// sub-block's output (`OLMo` 2).
    #[serde(default, skip_serializing_if = "NormPlacement::is_pre")]
    pub norm_placement: NormPlacement,
    pub mixer: MixerSpec,
    pub feed_forward: FeedForwardSpec,
}

/// Where a decoder block applies its norms.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormPlacement {
    /// `x + f(norm(x))`, optionally with a norm on `f`'s output (Gemma).
    #[default]
    Pre,
    /// `x + norm(f(x))`: the sub-blocks see the raw residual stream.
    Post,
    /// `x + attention(norm(x)) + feed_forward(norm(x))`: one norm feeds both
    /// sub-blocks, which read the same stream (Cohere).
    Parallel,
}

impl NormPlacement {
    #[must_use]
    pub const fn is_pre(&self) -> bool {
        matches!(self, Self::Pre)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NormalizationSpec {
    pub kind: NormalizationKind,
    pub epsilon: f64,
    /// Added to the learned weight before scaling: 1 for norms that scale by
    /// `1 + weight` (Gemma's), which no configuration declares.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub weight_offset: f64,
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde `skip_serializing_if` passes a reference"
)]
fn is_zero(value: &f64) -> bool {
    *value == 0.0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationKind {
    Rms,
    Layer,
}
