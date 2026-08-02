use crate::{
    layout::{PooledVisionConfig, SpatialMergeVisionConfig, VisionConfig},
    weights::{
        TensorCatalog, TensorReadiness, TensorRequirement, VisionTensorSchema, model_tensor_aliases,
    },
};

impl VisionTensorSchema {
    #[must_use]
    pub fn discover(config: &VisionConfig) -> Self {
        match config {
            VisionConfig::PooledEncoder(config) => pooled_encoder(config),
            VisionConfig::SpatialMergeEncoder(config) => spatial_merge_encoder(config),
        }
    }

    #[must_use]
    pub fn readiness(&self, catalog: &TensorCatalog) -> TensorReadiness {
        super::readiness(&self.requirements, catalog)
    }
}

fn pooled_encoder(config: &PooledVisionConfig) -> VisionTensorSchema {
    let mut requirements = vec![
        one("vision patch projection", "model.vision_tower.patch_embedder.input_proj.weight"),
        one(
            "vision position table",
            "model.vision_tower.patch_embedder.position_embedding_table",
        ),
        bound("vision text projection", "model.embed_vision.embedding_projection.weight"),
    ];
    if config.standardize {
        requirements.extend([
            one("vision standardization bias", "model.vision_tower.std_bias"),
            one("vision standardization scale", "model.vision_tower.std_scale"),
        ]);
    }
    for layer in 0..config.num_hidden_layers {
        push_pooled_layer(&mut requirements, layer, config.use_clipped_linears);
    }
    VisionTensorSchema { requirements }
}

fn push_pooled_layer(
    requirements: &mut Vec<TensorRequirement>,
    layer: usize,
    use_clipped_linears: bool,
) {
    let prefix = format!("model.vision_tower.encoder.layers.{layer}");
    for (label, suffix) in [
        ("vision input norm", "input_layernorm.weight"),
        ("vision post-attention norm", "post_attention_layernorm.weight"),
        ("vision pre-feed-forward norm", "pre_feedforward_layernorm.weight"),
        ("vision post-feed-forward norm", "post_feedforward_layernorm.weight"),
        ("vision query norm", "self_attn.q_norm.weight"),
        ("vision key norm", "self_attn.k_norm.weight"),
    ] {
        requirements.push(one(label, format!("{prefix}.{suffix}")));
    }
    for (label, suffix) in [
        ("vision query", "self_attn.q_proj"),
        ("vision key", "self_attn.k_proj"),
        ("vision value", "self_attn.v_proj"),
        ("vision attention output", "self_attn.o_proj"),
        ("vision MLP gate", "mlp.gate_proj"),
        ("vision MLP up", "mlp.up_proj"),
        ("vision MLP down", "mlp.down_proj"),
    ] {
        let projection = format!("{prefix}.{suffix}");
        requirements.push(linear(label, &projection));
        if use_clipped_linears {
            push_clipping(requirements, &projection);
        }
    }
}

fn push_clipping(requirements: &mut Vec<TensorRequirement>, prefix: &str) {
    for (label, suffix) in [
        ("vision clip input minimum", "input_min"),
        ("vision clip input maximum", "input_max"),
        ("vision clip output minimum", "output_min"),
        ("vision clip output maximum", "output_max"),
    ] {
        requirements.push(one(label, format!("{prefix}.{suffix}")));
    }
}

fn spatial_merge_encoder(config: &SpatialMergeVisionConfig) -> VisionTensorSchema {
    let mut requirements = vec![
        spatial_merge_one("vision patch weight", "patch_embed.proj.weight"),
        spatial_merge_one("vision patch bias", "patch_embed.proj.bias"),
        spatial_merge_one("vision position embedding", "pos_embed.weight"),
        spatial_merge_one("vision merger norm weight", "merger.norm.weight"),
        spatial_merge_one("vision merger norm bias", "merger.norm.bias"),
        spatial_merge_one("vision merger first weight", "merger.linear_fc1.weight"),
        spatial_merge_one("vision merger first bias", "merger.linear_fc1.bias"),
        spatial_merge_one("vision merger output weight", "merger.linear_fc2.weight"),
        spatial_merge_one("vision merger output bias", "merger.linear_fc2.bias"),
    ];
    for layer in 0..config.num_hidden_layers {
        push_spatial_merge_layer(&mut requirements, layer);
    }
    VisionTensorSchema { requirements }
}

fn push_spatial_merge_layer(requirements: &mut Vec<TensorRequirement>, layer: usize) {
    for (label, suffix) in [
        ("vision attention norm weight", "norm1.weight"),
        ("vision attention norm bias", "norm1.bias"),
        ("vision MLP norm weight", "norm2.weight"),
        ("vision MLP norm bias", "norm2.bias"),
        ("vision QKV weight", "attn.qkv.weight"),
        ("vision QKV bias", "attn.qkv.bias"),
        ("vision attention output weight", "attn.proj.weight"),
        ("vision attention output bias", "attn.proj.bias"),
        ("vision MLP first weight", "mlp.linear_fc1.weight"),
        ("vision MLP first bias", "mlp.linear_fc1.bias"),
        ("vision MLP output weight", "mlp.linear_fc2.weight"),
        ("vision MLP output bias", "mlp.linear_fc2.bias"),
    ] {
        requirements.push(spatial_merge_one(label, format!("blocks.{layer}.{suffix}")));
    }
}

fn spatial_merge_one(label: &str, suffix: impl AsRef<str>) -> TensorRequirement {
    let suffix = suffix.as_ref();
    TensorRequirement::any(
        label,
        vec![format!("model.visual.{suffix}"), format!("vision_tower.{suffix}")],
    )
}

fn one(label: &str, name: impl Into<String>) -> TensorRequirement {
    TensorRequirement::any(label, model_tensor_aliases(name))
}

fn bound(label: &str, name: impl Into<String>) -> TensorRequirement {
    TensorRequirement::bound(label, model_tensor_aliases(name))
}

fn linear(label: &str, prefix: &str) -> TensorRequirement {
    let mut aliases = model_tensor_aliases(format!("{prefix}.linear.weight"));
    aliases.extend(model_tensor_aliases(format!("{prefix}.weight")));
    TensorRequirement::any(label, aliases)
}

#[cfg(test)]
#[path = "vision/tests.rs"]
mod tests;
