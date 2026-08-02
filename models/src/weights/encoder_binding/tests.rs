use std::path::PathBuf;

use super::*;
use crate::{
    layout::{EncoderRopeScaling, NormKind},
    weights::TensorInfo,
};

#[test]
fn binds_encoder_roles_and_dense_storage() -> Result<()> {
    let config = config();
    let catalog = catalog(&config, "F16");
    let plan = EncoderBindingPlan::discover(&config, &catalog)?;

    assert_eq!(plan.tensors.len(), 10);
    assert!(plan.tensors.iter().any(|binding| {
        binding.role
            == EncoderTensorRole::Layer {
                index: 0,
                tensor: EncoderLayerTensorRole::Qkv,
            }
            && matches!(binding.storage, TensorStorage::Dense { ref dtype, .. } if dtype == "F16")
    }));
    Ok(())
}

#[test]
fn rejects_an_encoder_shape_mismatch() {
    let config = config();
    let mut catalog = catalog(&config, "F16");
    if let Some(tensor) = catalog.tensors.first_mut() {
        tensor.shape = vec![999];
    }
    assert!(EncoderBindingPlan::discover(&config, &catalog).is_err());
}

#[test]
fn rejects_an_encoder_bias_with_a_different_dtype() -> Result<()> {
    let config = config();
    let mut catalog = catalog(&config, "F16");
    let Some(bias) = catalog.tensors.iter_mut().find(|tensor| tensor.name == "classifier.bias")
    else {
        return Err(ModelsError::InvalidConfig("classifier bias fixture is missing".into()));
    };
    bias.dtype = "BF16".into();

    assert!(EncoderBindingPlan::discover(&config, &catalog).is_err());
    Ok(())
}

fn config() -> EncoderConfig {
    EncoderConfig {
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        head_dim: 4,
        vocab_size: 32,
        max_position_embeddings: 64,
        layer_norm_eps: 1e-5,
        hidden_activation: "gelu".into(),
        position_embedding: EncoderPositionEmbedding::Rope,
        rope_theta: Some(10_000.0),
        rope_scaling: Some(EncoderRopeScaling::Ntk { factor: 1.0, mixed_b: None }),
        norm: NormKind::LayerNorm,
        type_vocab_size: 0,
        packed_qkv: true,
        num_labels: 1,
    }
}

fn catalog(config: &EncoderConfig, dtype: &str) -> TensorCatalog {
    let schema = crate::weights::EncoderTensorSchema::discover(
        config,
        &TensorCatalog { tensors: Vec::new() },
    );
    let mut tensors = schema
        .requirements
        .into_iter()
        .map(|requirement| {
            let name = requirement.aliases[0].clone();
            TensorInfo {
                shape: shape(&name, config),
                name,
                file: PathBuf::from("model.safetensors"),
                dtype: dtype.into(),
                data_start: 0,
                data_offsets: [0, 0],
            }
        })
        .collect::<Vec<_>>();
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    TensorCatalog { tensors }
}

fn shape(name: &str, config: &EncoderConfig) -> Vec<usize> {
    let hidden = config.hidden_size;
    if name == "new.embeddings.word_embeddings.weight" {
        vec![config.vocab_size, hidden]
    } else if name.ends_with("qkv_proj.weight") {
        vec![hidden * 3, hidden]
    } else if name.ends_with("qkv_proj.bias") {
        vec![hidden * 3]
    } else if name.ends_with("up_gate_proj.weight") {
        vec![config.intermediate_size * 2, hidden]
    } else if name.ends_with("down_proj.weight") {
        vec![hidden, config.intermediate_size]
    } else if name == "classifier.weight" {
        vec![config.num_labels, hidden]
    } else if name == "classifier.bias" {
        vec![config.num_labels]
    } else if name.ends_with(".weight") && (name.contains("dense") || name.contains("o_proj")) {
        vec![hidden, hidden]
    } else if name
        .rsplit_once('.')
        .is_some_and(|(_, suffix)| matches!(suffix, "weight" | "bias"))
    {
        vec![hidden]
    } else {
        Vec::new()
    }
}
