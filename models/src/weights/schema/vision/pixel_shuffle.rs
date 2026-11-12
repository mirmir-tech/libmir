//! Tensors of the pixel-shuffle encoder: the patch generator (projection,
//! position table, prefix tokens), pre-norm transformer blocks with biased
//! linears,
//! the optional final layer norm, and the three-stage projector.

use crate::{
    layout::PixelShuffleVisionConfig,
    weights::{TensorRequirement, VisionTensorSchema},
};

const ENCODER: &str = "vision_model.radio_model.model";

pub(super) fn pixel_shuffle_encoder(config: &PixelShuffleVisionConfig) -> VisionTensorSchema {
    let mut requirements = vec![
        encoder("vision patch projection", "patch_generator.embedder.weight"),
        encoder("vision position table", "patch_generator.pos_embed"),
        encoder("vision prefix tokens", "patch_generator.cls_token.token"),
        named("vision projector norm", "mlp1.0.weight"),
        named("vision projector first weight", "mlp1.1.weight"),
        named("vision projector output weight", "mlp1.3.weight"),
    ];
    if config.final_layer_norm {
        requirements.push(named(
            "vision final norm weight",
            "vision_projector.vision_final_layernorm.weight",
        ));
        requirements
            .push(named("vision final norm bias", "vision_projector.vision_final_layernorm.bias"));
    }
    for layer in 0..config.num_hidden_layers {
        for (label, suffix) in [
            ("vision attention norm weight", "norm1.weight"),
            ("vision attention norm bias", "norm1.bias"),
            ("vision MLP norm weight", "norm2.weight"),
            ("vision MLP norm bias", "norm2.bias"),
            ("vision QKV weight", "attn.qkv.weight"),
            ("vision QKV bias", "attn.qkv.bias"),
            ("vision attention output weight", "attn.proj.weight"),
            ("vision attention output bias", "attn.proj.bias"),
            ("vision MLP first weight", "mlp.fc1.weight"),
            ("vision MLP first bias", "mlp.fc1.bias"),
            ("vision MLP output weight", "mlp.fc2.weight"),
            ("vision MLP output bias", "mlp.fc2.bias"),
        ] {
            requirements.push(encoder(label, &format!("blocks.{layer}.{suffix}")));
        }
    }
    VisionTensorSchema { requirements }
}

/// The tensor names one requirement of the contract resolves to.
#[must_use]
pub fn encoder_tensor(suffix: &str) -> String {
    format!("{ENCODER}.{suffix}")
}

fn encoder(label: &str, suffix: &str) -> TensorRequirement {
    TensorRequirement::any(label, vec![encoder_tensor(suffix)])
}

fn named(label: &str, name: &str) -> TensorRequirement {
    TensorRequirement::any(label, vec![name.to_owned()])
}
