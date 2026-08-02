use std::path::PathBuf;

use foundation::model::BackendTarget;
use models::weights::{TensorCatalog, TensorInfo};
use serde_json::json;

use super::*;
use crate::{AdmissionCheckKind, AdmissionStatus};

mod clamped_routed_dense;
mod routed_dense;
mod shared_routed_dense;
mod vision;

#[test]
fn builds_the_same_typed_contract_from_remote_headers() -> Result<()> {
    let config = json!({
        "model_type": "mistral",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 64,
        "hidden_act": "silu"
    });
    let catalog = dense_catalog();

    let contract = RemoteModelContract::inspect_generation(&config, &catalog)?;

    assert_eq!(contract.execution().map(|execution| execution.bindings.tensors.len()), Some(12));
    assert_eq!(contract.checkpoint_encoding().label(), "Dense BF16");
    assert_eq!(contract.admission(BackendTarget::Metal).status, AdmissionStatus::Supported);
    let cuda = contract.admission(BackendTarget::Cuda);
    assert_eq!(cuda.status, AdmissionStatus::Supported);
    assert!(cuda.checks.iter().any(|check| {
        check.kind == AdmissionCheckKind::Architecture && check.status == AdmissionStatus::Supported
    }));
    Ok(())
}

#[test]
fn admits_dense_decoder_storage_converted_by_cuda_on_load() -> Result<()> {
    for dtype in ["F16", "F32"] {
        let contract = RemoteModelContract::inspect_generation(
            &decoder_config(),
            &dense_catalog_with_dtype(dtype),
        )?;

        assert_eq!(contract.admission(BackendTarget::Metal).status, AdmissionStatus::Supported);
        let cuda = contract.admission(BackendTarget::Cuda);
        assert_eq!(cuda.status, AdmissionStatus::Supported);
        assert!(cuda.checks.iter().any(|check| {
            check.kind == AdmissionCheckKind::Dense
                && check.status == AdmissionStatus::Supported
                && check.detail.contains("converts dense storage")
        }));
    }
    Ok(())
}

#[test]
fn discovers_remote_sentence_transformers_task() -> Result<()> {
    let config = decoder_config();
    let catalog = dense_catalog();
    let modules = json!([
        {"path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
        {"path": "2_Normalize", "type": "sentence_transformers.models.Normalize"}
    ]);
    let pooling = json!({
        "word_embedding_dimension": 32,
        "pooling_mode_cls_token": false,
        "pooling_mode_mean_tokens": false,
        "pooling_mode_lasttoken": true
    });

    let contract = RemoteModelContract::inspect(
        &config,
        &catalog,
        RemoteTaskMetadata {
            modules: Some(&modules),
            pooling: Some(&pooling),
            ..Default::default()
        },
    )?
    .ok_or_else(|| Error::Model(models::ModelsError::InvalidConfig("task missing".into())))?;

    assert!(matches!(contract.task(), TaskExecutionPlan::Embedding { .. }));
    assert_eq!(contract.admission(BackendTarget::Metal).status, AdmissionStatus::Supported);
    assert_eq!(contract.admission(BackendTarget::Cuda).status, AdmissionStatus::Supported);

    for dtype in ["F16", "F32"] {
        let converted = RemoteModelContract::inspect(
            &config,
            &dense_catalog_with_dtype(dtype),
            RemoteTaskMetadata {
                modules: Some(&modules),
                pooling: Some(&pooling),
                ..Default::default()
            },
        )?
        .ok_or_else(|| Error::Model(models::ModelsError::InvalidConfig("task missing".into())))?;
        let cuda = converted.admission(BackendTarget::Cuda);
        assert_eq!(cuda.status, AdmissionStatus::Supported);
        assert!(cuda.checks.iter().any(|check| check.detail.contains("converts dense storage")));
    }

    Ok(())
}

#[test]
fn discovers_remote_sequence_scoring_task() -> Result<()> {
    let config = json!({
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 0,
        "num_attention_heads": 4,
        "vocab_size": 64,
        "max_position_embeddings": 512,
        "layer_norm_eps": 1e-5,
        "hidden_act": "gelu",
        "position_embedding_type": "rope",
        "layer_norm_type": "layer_norm",
        "pack_qkv": true,
        "type_vocab_size": 2,
        "rope_scaling": {"type": "ntk", "factor": 1.0},
        "num_labels": 1
    });
    let catalog = sequence_scoring_catalog("F16");

    let contract = RemoteModelContract::inspect(&config, &catalog, RemoteTaskMetadata::default())?
        .ok_or_else(|| Error::Model(models::ModelsError::InvalidConfig("task missing".into())))?;

    assert!(matches!(contract.task(), TaskExecutionPlan::SequenceScoring { .. }));
    assert!(contract.execution().is_none());
    assert_eq!(contract.checkpoint_encoding().label(), "Dense F16");
    assert_eq!(contract.admission(BackendTarget::Metal).status, AdmissionStatus::Supported);
    assert_eq!(contract.admission(BackendTarget::Cuda).status, AdmissionStatus::Supported);

    let bf16 = RemoteModelContract::inspect(
        &config,
        &sequence_scoring_catalog("BF16"),
        RemoteTaskMetadata::default(),
    )?
    .ok_or_else(|| Error::Model(models::ModelsError::InvalidConfig("task missing".into())))?;
    assert_eq!(bf16.admission(BackendTarget::Metal).status, AdmissionStatus::Supported);
    assert_eq!(bf16.admission(BackendTarget::Cuda).status, AdmissionStatus::Unsupported);

    let mut metal_only_config = config;
    metal_only_config["type_vocab_size"] = json!(0);
    let mut metal_only_catalog = catalog;
    metal_only_catalog
        .tensors
        .retain(|tensor| tensor.name != "new.embeddings.token_type_embeddings.weight");
    let metal_only = RemoteModelContract::inspect(
        &metal_only_config,
        &metal_only_catalog,
        RemoteTaskMetadata::default(),
    )?
    .ok_or_else(|| Error::Model(models::ModelsError::InvalidConfig("task missing".into())))?;
    assert_eq!(metal_only.admission(BackendTarget::Metal).status, AdmissionStatus::Supported);
    assert_eq!(metal_only.admission(BackendTarget::Cuda).status, AdmissionStatus::Unsupported);
    Ok(())
}

fn sequence_scoring_catalog(dtype: &str) -> TensorCatalog {
    TensorCatalog {
        tensors: [
            "new.embeddings.word_embeddings.weight",
            "new.embeddings.token_type_embeddings.weight",
            "new.embeddings.LayerNorm.weight",
            "new.embeddings.LayerNorm.bias",
            "new.encoder.layer.0.attention.qkv_proj.weight",
            "new.pooler.dense.weight",
            "new.pooler.dense.bias",
            "classifier.weight",
            "classifier.bias",
        ]
        .into_iter()
        .map(|name| {
            let shape = match name {
                "new.embeddings.word_embeddings.weight" => vec![64, 32],
                "new.embeddings.token_type_embeddings.weight" => vec![2, 32],
                "new.pooler.dense.weight" => vec![32, 32],
                "classifier.weight" => vec![1, 32],
                "classifier.bias" => vec![1],
                _ => vec![32],
            };
            tensor_with_dtype(name, shape, dtype)
        })
        .collect(),
    }
}

fn decoder_config() -> serde_json::Value {
    json!({
        "model_type": "mistral",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 64,
        "hidden_act": "silu"
    })
}

pub(super) fn dense_catalog() -> TensorCatalog {
    dense_catalog_with_dtype("BF16")
}

fn dense_catalog_with_dtype(dtype: &str) -> TensorCatalog {
    let mut tensors = vec![
        tensor_with_dtype("model.embed_tokens.weight", vec![64, 32], dtype),
        tensor_with_dtype("model.norm.weight", vec![32], dtype),
        tensor_with_dtype("lm_head.weight", vec![64, 32], dtype),
        tensor_with_dtype("model.layers.0.input_layernorm.weight", vec![32], dtype),
        tensor_with_dtype("model.layers.0.self_attn.q_proj.weight", vec![32, 32], dtype),
        tensor_with_dtype("model.layers.0.self_attn.k_proj.weight", vec![16, 32], dtype),
        tensor_with_dtype("model.layers.0.self_attn.v_proj.weight", vec![16, 32], dtype),
        tensor_with_dtype("model.layers.0.self_attn.o_proj.weight", vec![32, 32], dtype),
        tensor_with_dtype("model.layers.0.post_attention_layernorm.weight", vec![32], dtype),
        tensor_with_dtype("model.layers.0.mlp.gate_proj.weight", vec![64, 32], dtype),
        tensor_with_dtype("model.layers.0.mlp.up_proj.weight", vec![64, 32], dtype),
        tensor_with_dtype("model.layers.0.mlp.down_proj.weight", vec![32, 64], dtype),
    ];
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    TensorCatalog { tensors }
}

pub(super) fn tensor(name: &str, shape: Vec<usize>) -> TensorInfo {
    tensor_with_dtype(name, shape, "BF16")
}

fn tensor_with_dtype(name: &str, shape: Vec<usize>, dtype: &str) -> TensorInfo {
    TensorInfo {
        name: name.to_owned(),
        file: PathBuf::from("remote.safetensors"),
        dtype: dtype.to_owned(),
        shape,
        data_start: 0,
        data_offsets: [0, 0],
    }
}
