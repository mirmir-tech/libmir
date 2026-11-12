//! The pooled encoder contract: a fixed patch grid, spatial pooling and a
//! text projection.

use serde_json::Value;

use super::super::parse::{
    bool_field, float_field, has_fields, object, optional_usize_field, string_field, u32_field,
    usize_field,
};
use crate::error::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct PooledVisionConfig {
    pub hidden_size: usize,
    pub output_hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub position_embedding_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub hidden_activation: String,
    pub use_clipped_linears: bool,
    pub standardize: bool,
    pub image_token_id: u32,
    pub image_begin_token_id: u32,
    pub image_end_token_id: u32,
    pub soft_tokens_per_image: usize,
    pub bidirectional_image_attention: bool,
}

pub(super) fn is_pooled_encoder(root: &Value, vision: &Value) -> bool {
    has_fields(
        vision,
        &[
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "patch_size",
            "pooling_kernel_size",
            "position_embedding_size",
            "rms_norm_eps",
            "hidden_activation",
        ],
    ) && has_fields(
        root,
        &[
            "text_config",
            "image_token_id",
            "boi_token_id",
            "eoi_token_id",
            "vision_soft_tokens_per_image",
        ],
    )
}

pub(super) fn parse_pooled(root: &Value, vision: &Value) -> Result<PooledVisionConfig> {
    let text = object(root, "text_config")?;
    let num_attention_heads = usize_field(vision, "num_attention_heads")?;
    let hidden_size = usize_field(vision, "hidden_size")?;
    let head_dim = optional_usize_field(vision, "head_dim")?
        .unwrap_or_else(|| hidden_size / num_attention_heads);
    let rope_theta = vision
        .get("rope_parameters")
        .and_then(|value| value.get("rope_theta"))
        .and_then(Value::as_f64)
        .unwrap_or(100.0);
    Ok(PooledVisionConfig {
        hidden_size,
        output_hidden_size: usize_field(text, "hidden_size")?,
        intermediate_size: usize_field(vision, "intermediate_size")?,
        num_hidden_layers: usize_field(vision, "num_hidden_layers")?,
        num_attention_heads,
        num_key_value_heads: optional_usize_field(vision, "num_key_value_heads")?
            .unwrap_or(num_attention_heads),
        head_dim,
        patch_size: usize_field(vision, "patch_size")?,
        pooling_kernel_size: usize_field(vision, "pooling_kernel_size")?,
        position_embedding_size: usize_field(vision, "position_embedding_size")?,
        rms_norm_eps: float_field(vision, "rms_norm_eps")?,
        rope_theta,
        hidden_activation: string_field(vision, "hidden_activation")?,
        use_clipped_linears: bool_field(vision, "use_clipped_linears", false)?,
        standardize: bool_field(vision, "standardize", false)?,
        image_token_id: u32_field(root, "image_token_id")?,
        image_begin_token_id: u32_field(root, "boi_token_id")?,
        image_end_token_id: u32_field(root, "eoi_token_id")?,
        soft_tokens_per_image: usize_field(root, "vision_soft_tokens_per_image")?,
        bidirectional_image_attention: text
            .get("use_bidirectional_attention")
            .and_then(Value::as_str)
            == Some("vision"),
    })
}
