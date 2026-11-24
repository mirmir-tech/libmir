//! Tensors of the per-layer inputs of Gemma 4's E models.

use serde::{Deserialize, Serialize};

/// The model's tensors that form every layer's input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerLayerEmbeddingRole {
    /// `embed_tokens_per_layer`: every layer's slice of a second token
    /// embedding.
    Embedding,
    /// `per_layer_model_projection`: projects the scaled token embeddings to
    /// every layer's slice.
    ModelProjection,
    /// `per_layer_projection_norm`: normalizes each slice of that projection.
    ProjectionNorm,
}

/// A layer's tensors that gate its input into its output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerLayerInputRole {
    /// `per_layer_input_gate`: from the hidden state to the input width.
    Gate,
    /// `per_layer_projection`: from the gated input back to the hidden width.
    Projection,
    /// `post_per_layer_input_norm`: normalizes that projection before the
    /// residual add.
    OutputNorm,
}
