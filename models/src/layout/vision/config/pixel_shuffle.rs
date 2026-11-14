//! The pixel-shuffle encoder contract: a vision transformer whose learned
//! prefix tokens
//! precede the patches, an absolute position grid resized to each image,
//! a pixel shuffle that folds neighbouring patches into channels, and an
//! RMS-normalized squared-rectifier projector into the decoder's hidden size.

use serde_json::Value;

use super::super::parse::{
    float_field, has_fields, invalid, object, optional_usize_field, string_field, u32_field,
    usize_field,
};
use crate::error::Result;

/// The projector's RMS norm epsilon, which configurations of this contract
/// do not declare.
const PROJECTOR_NORM_EPS: f64 = 1e-5;

#[derive(Debug, Clone, PartialEq)]
pub struct PixelShuffleVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub patch_size: usize,
    /// Learned summary and register tokens before the patches.
    pub prefix_tokens: usize,
    /// Side of the square position table, in patches.
    pub position_grid: usize,
    pub layer_norm_eps: f64,
    pub hidden_activation: String,
    /// Patches folded into one token along each axis.
    pub shuffle_factor: usize,
    pub projector_hidden_size: usize,
    pub projector_norm_eps: f64,
    pub output_hidden_size: usize,
    /// A layer norm over the encoder output before the shuffle.
    pub final_layer_norm: bool,
    pub image_token_id: u32,
    pub image_tags: ImageTags,
}

/// How the image's tokens are bracketed in the prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageTags {
    /// `<img>` … `</img>` around the image tokens.
    Img,
}

impl ImageTags {
    #[must_use]
    pub const fn start(self) -> &'static str {
        match self {
            Self::Img => "<img>",
        }
    }

    #[must_use]
    pub const fn end(self) -> &'static str {
        match self {
            Self::Img => "</img>",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "internvl" => Ok(Self::Img),
            other => Err(invalid(format!("unsupported image tag type {other}"))),
        }
    }
}

pub(super) fn is_pixel_shuffle_encoder(root: &Value, vision: &Value) -> bool {
    has_fields(
        vision,
        &[
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "patch_size",
            "mlp_ratio",
            "num_cls_tokens",
            "num_registers",
            "max_img_size",
            "layer_norm_eps",
            "hidden_act",
        ],
    ) && has_fields(
        root,
        &[
            "llm_config",
            "downsample_ratio",
            "ps_version",
            "vit_hidden_size",
            "projector_hidden_size",
            "img_context_token_id",
            "image_tag_type",
        ],
    )
}

pub(super) fn parse_pixel_shuffle(
    root: &Value,
    vision: &Value,
) -> Result<PixelShuffleVisionConfig> {
    let decoder = object(root, "llm_config")?;
    let hidden_size = usize_field(vision, "hidden_size")?;
    if usize_field(root, "vit_hidden_size")? != hidden_size {
        return Err(invalid("vit_hidden_size differs from the vision hidden size"));
    }
    // Only the shuffle that swaps the axes back keeps the image upright.
    if string_field(root, "ps_version")? != "v2" {
        return Err(invalid("unsupported pixel-shuffle version"));
    }
    // A layer scale other than one would need tensors the contract does not
    // carry.
    if vision
        .get("layerscale_value")
        .and_then(Value::as_f64)
        .is_some_and(|scale| (scale - 1.0).abs() > f64::EPSILON)
    {
        return Err(invalid("unsupported layer scale without scale tensors"));
    }
    let patch_size = usize_field(vision, "patch_size")?;
    let max_img_size = usize_field(vision, "max_img_size")?;
    if patch_size == 0 || max_img_size % patch_size != 0 {
        return Err(invalid("max_img_size is not a whole number of patches"));
    }
    Ok(PixelShuffleVisionConfig {
        hidden_size,
        intermediate_size: intermediate_size(vision, hidden_size)?,
        num_hidden_layers: usize_field(vision, "num_hidden_layers")?,
        num_attention_heads: usize_field(vision, "num_attention_heads")?,
        num_channels: optional_usize_field(vision, "num_channels")?.unwrap_or(3),
        patch_size,
        prefix_tokens: usize_field(vision, "num_cls_tokens")?
            .checked_add(usize_field(vision, "num_registers")?)
            .ok_or_else(|| invalid("vision prefix token count overflows"))?,
        position_grid: max_img_size / patch_size,
        layer_norm_eps: float_field(vision, "layer_norm_eps")?,
        hidden_activation: string_field(vision, "hidden_act")?,
        shuffle_factor: shuffle_factor(float_field(root, "downsample_ratio")?)?,
        projector_hidden_size: usize_field(root, "projector_hidden_size")?,
        projector_norm_eps: PROJECTOR_NORM_EPS,
        output_hidden_size: usize_field(decoder, "hidden_size")?,
        // The encoder shares the decoder's block configuration: a decoder
        // with prediction layers ends every block stack in a layer norm.
        final_layer_norm: optional_usize_field(decoder, "num_nextn_predict_layers")?.unwrap_or(0)
            > 0,
        image_token_id: u32_field(root, "img_context_token_id")?,
        image_tags: ImageTags::parse(&string_field(root, "image_tag_type")?)?,
    })
}

fn intermediate_size(vision: &Value, hidden_size: usize) -> Result<usize> {
    let ratio = float_field(vision, "mlp_ratio")?;
    let size = f64::from(u32::try_from(hidden_size)?) * ratio;
    if size.fract() != 0.0 || size <= 0.0 {
        return Err(invalid("mlp_ratio does not give a whole intermediate size"));
    }
    Ok(size.to_string().parse()?)
}

/// `downsample_ratio` 0.5 folds 2×2 patches into one token.
pub(in crate::layout::vision) fn shuffle_factor(ratio: f64) -> Result<usize> {
    let factor = ratio.recip();
    if !(factor >= 1.0 && factor.fract() == 0.0) {
        return Err(invalid("downsample_ratio is not the inverse of a whole factor"));
    }
    Ok(factor.to_string().parse()?)
}
