use std::{fs, path::PathBuf};

use serde_json::json;

use super::*;
use crate::{
    error::Result,
    layout::{DecoderConfig, ModelLayout},
    weights::{TensorCatalog, TensorInfo},
};

#[test]
fn semantic_discovery_does_not_depend_on_model_type() -> Result<()> {
    let named = DecoderConfig::from_value(&configuration("gpt_oss", &["GptOssForCausalLM"]))?;
    let unrelated = DecoderConfig::from_value(&configuration(
        "unrelated",
        &["MisleadingRepositoryPrefixModel"],
    ))?;
    let tensors = attention_sink_catalog();

    let named = SemanticModelSpec::discover(&named, &tensors)?;
    let unrelated = SemanticModelSpec::discover(&unrelated, &tensors)?;

    assert_eq!(named, unrelated);
    assert!(matches!(
        named.decoder.layers[0].mixer,
        MixerSpec::SoftmaxAttention(AttentionSpec { sinks: true, .. })
    ));
    Ok(())
}

#[test]
fn key_equals_value_applies_only_to_full_attention() -> Result<()> {
    let mut config = configuration("hybrid", &[]);
    config["attention_k_eq_v"] = json!(true);
    config["global_head_dim"] = json!(16);
    config["num_global_key_value_heads"] = json!(1);
    let decoder = DecoderConfig::from_value(&config)?;
    let spec = SemanticModelSpec::discover(&decoder, &TensorCatalog { tensors: Vec::new() })?;

    let relations = spec
        .decoder
        .layers
        .iter()
        .map(|layer| match &layer.mixer {
            MixerSpec::SoftmaxAttention(attention) => Some(attention.key_value_relation),
            MixerSpec::LinearAttention(_) => None,
        })
        .collect::<Option<Vec<_>>>();
    assert_eq!(
        relations,
        Some(vec![KeyValueRelation::Separate, KeyValueRelation::KeyEqualsValue])
    );
    Ok(())
}

#[test]
fn toml_sidecar_round_trips_complete_semantics() -> Result<()> {
    let decoder = DecoderConfig::from_value(&configuration("unrelated", &[]))?;
    let spec = SemanticModelSpec::discover(&decoder, &attention_sink_catalog())?;
    let encoded = spec.to_toml()?;
    assert!(encoded.contains("schema_version = 1"));
    assert!(encoded.contains("[[decoder.layers]]"));

    let path = temp_path();
    fs::write(&path, encoded)?;
    let restored = sidecar::read(&path)?;
    let _removed = fs::remove_file(path);

    assert_eq!(restored, spec);
    Ok(())
}

#[test]
fn config_and_sidecar_paths_produce_equal_semantics() -> Result<()> {
    let root = temp_dir("misleading-gpt-oss-repository-prefix");
    fs::create_dir_all(&root)?;
    fs::write(
        root.join("config.json"),
        serde_json::to_vec(&configuration("renamed", &["UnrelatedArchitecture"]))?,
    )?;
    let layout = ModelLayout::inspect(&root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = attention_sink_catalog();
    let derived = SemanticModelSpec::from_layout(&layout, &decoder, &catalog)?;

    fs::write(root.join("mir-model-spec.toml"), derived.to_toml()?)?;
    fs::write(
        root.join("config.json"),
        serde_json::to_vec(&configuration("another_name", &["AnotherArchitecture"]))?,
    )?;
    let sidecar_layout = ModelLayout::inspect(&root)?;
    let sidecar_decoder = DecoderConfig::from_layout(&sidecar_layout)?;
    let from_sidecar = SemanticModelSpec::from_layout(&sidecar_layout, &sidecar_decoder, &catalog)?;
    fs::remove_dir_all(root)?;

    assert_eq!(from_sidecar, derived);
    Ok(())
}

fn configuration(model_type: &str, architectures: &[&str]) -> serde_json::Value {
    json!({
        "model_type": model_type,
        "architectures": architectures,
        "hidden_size": 32,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "hidden_act": "silu",
        "attention_bias": true,
        "swiglu_limit": 7.0,
        "sliding_window": 16,
        "layer_types": ["sliding_attention", "full_attention"],
        "rope_theta": 150_000.0,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 4.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "original_max_position_embeddings": 32
        }
    })
}

fn attention_sink_catalog() -> TensorCatalog {
    catalog(&["model.layers.0.self_attn.sinks", "model.layers.1.self_attn.sinks"])
}

fn catalog(names: &[&str]) -> TensorCatalog {
    TensorCatalog {
        tensors: names
            .iter()
            .map(|name| TensorInfo {
                name: (*name).to_owned(),
                file: PathBuf::new(),
                dtype: "F32".into(),
                shape: vec![4],
                data_start: 0,
                data_offsets: [0, 0],
            })
            .collect(),
    }
}

fn temp_path() -> PathBuf {
    std::env::temp_dir().join(format!("mir-model-spec-{}.toml", std::process::id()))
}

fn temp_dir(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!("mir-model-spec-{}-{label}", std::process::id()))
}
