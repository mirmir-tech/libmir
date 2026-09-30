use serde_json::{Value, json};

use super::{ModernBertAttention, ModernBertConfig};
use crate::Result;

fn multilingual() -> Value {
    json!({
        "model_type": "modernbert",
        "hidden_size": 768,
        "intermediate_size": 1152,
        "num_hidden_layers": 4,
        "num_attention_heads": 12,
        "vocab_size": 256_000,
        "max_position_embeddings": 8192,
        "norm_eps": 1e-5,
        "hidden_activation": "gelu",
        "local_attention": 128,
        "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
        "rope_parameters": {
            "full_attention": {"rope_theta": 160_000, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 10_000.0, "rope_type": "default"}
        },
        "attention_bias": false,
        "mlp_bias": false,
        "norm_bias": false
    })
}

#[test]
fn reads_per_layer_attention_and_rotary_bases() -> Result<()> {
    let config = ModernBertConfig::from_json(&multilingual().to_string())?;
    let local = ModernBertAttention::Local { radius: 64 };
    assert_eq!(config.head_dim, 64);
    assert_eq!(
        config.layers,
        vec![ModernBertAttention::Global, local, local, ModernBertAttention::Global]
    );
    assert!((config.rope_theta(ModernBertAttention::Global) - 160_000.0).abs() < f64::EPSILON);
    assert!((config.rope_theta(local) - 10_000.0).abs() < f64::EPSILON);
    Ok(())
}

#[test]
fn derives_layers_and_thetas_from_the_older_config_form() -> Result<()> {
    let mut value = multilingual();
    if let Some(object) = value.as_object_mut() {
        let _layers = object.remove("layer_types");
        let _rope = object.remove("rope_parameters");
        let _every = object.insert("global_attn_every_n_layers".into(), json!(3));
        let _global = object.insert("global_rope_theta".into(), json!(160_000.0));
        let _local = object.insert("local_rope_theta".into(), json!(10_000.0));
    }
    let config = ModernBertConfig::from_json(&value.to_string())?;
    assert_eq!(config.layers[3], ModernBertAttention::Global);
    assert_eq!(config.layers[1], ModernBertAttention::Local { radius: 64 });
    Ok(())
}

#[test]
fn rejects_biases_and_missing_rotary_bases() {
    let mut biased = multilingual();
    biased["norm_bias"] = json!(true);
    assert!(ModernBertConfig::from_json(&biased.to_string()).is_err());

    let mut unrotated = multilingual();
    unrotated["rope_parameters"] = json!({});
    assert!(ModernBertConfig::from_json(&unrotated.to_string()).is_err());
}
