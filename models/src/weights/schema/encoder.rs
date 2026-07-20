use crate::{
    layout::{EncoderConfig, EncoderPositionEmbedding},
    weights::{EncoderTensorSchema, TensorCatalog, TensorRequirement},
};

pub(super) fn discover(config: &EncoderConfig, _catalog: &TensorCatalog) -> EncoderTensorSchema {
    let mut requirements = vec![
        one("token embeddings", "new.embeddings.word_embeddings.weight"),
        one("embedding norm weight", "new.embeddings.LayerNorm.weight"),
        one("embedding norm bias", "new.embeddings.LayerNorm.bias"),
    ];
    if config.type_vocab_size > 0 {
        requirements
            .push(one("token type embeddings", "new.embeddings.token_type_embeddings.weight"));
    }
    if config.position_embedding == EncoderPositionEmbedding::Absolute {
        requirements.push(one("position embeddings", "new.embeddings.position_embeddings.weight"));
    }
    for layer in 0..config.num_hidden_layers {
        push_layer(&mut requirements, layer, config.packed_qkv);
    }
    requirements.extend([
        one("pooler weight", "new.pooler.dense.weight"),
        one("pooler bias", "new.pooler.dense.bias"),
        one("classifier weight", "classifier.weight"),
        one("classifier bias", "classifier.bias"),
    ]);
    EncoderTensorSchema { requirements }
}

fn push_layer(requirements: &mut Vec<TensorRequirement>, layer: usize, packed_qkv: bool) {
    let prefix = format!("new.encoder.layer.{layer}");
    if packed_qkv {
        requirements.extend([
            child("packed QKV weight", &prefix, "attention.qkv_proj.weight"),
            child("packed QKV bias", &prefix, "attention.qkv_proj.bias"),
        ]);
    } else {
        for projection in ["q", "k", "v"] {
            requirements.extend([
                child(
                    &format!("attention {projection} weight"),
                    &prefix,
                    &format!("attention.{projection}_proj.weight"),
                ),
                child(
                    &format!("attention {projection} bias"),
                    &prefix,
                    &format!("attention.{projection}_proj.bias"),
                ),
            ]);
        }
    }
    requirements.extend([
        child("attention output weight", &prefix, "attention.o_proj.weight"),
        child("attention output bias", &prefix, "attention.o_proj.bias"),
        child("attention norm weight", &prefix, "attn_ln.weight"),
        child("attention norm bias", &prefix, "attn_ln.bias"),
        child("MLP up/gate weight", &prefix, "mlp.up_gate_proj.weight"),
        child("MLP down weight", &prefix, "mlp.down_proj.weight"),
        child("MLP down bias", &prefix, "mlp.down_proj.bias"),
        child("MLP norm weight", &prefix, "mlp_ln.weight"),
        child("MLP norm bias", &prefix, "mlp_ln.bias"),
    ]);
}

fn child(label: &str, prefix: &str, suffix: &str) -> TensorRequirement {
    one(label, format!("{prefix}.{suffix}"))
}

fn one(label: &str, name: impl Into<String>) -> TensorRequirement {
    TensorRequirement::any(label, vec![name.into()])
}
