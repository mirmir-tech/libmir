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
