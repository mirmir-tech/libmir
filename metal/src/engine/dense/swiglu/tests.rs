use std::{fs, path::Path};

use models::{
    execution::DecoderExecutionContract,
    layout::{DecoderConfig, ModelLayout},
    weights::TensorCatalog,
};
use serde_json::{Map, Value, json};

use super::DenseSwiGluModel;
use crate::engine::{Array, ModelTensors, Result, Stream, lowering};

#[test]
fn executes_complete_dense_models_without_affine_group_size() -> Result<()> {
    for dtype in ["F32", "F16", "BF16"] {
        execute_model(dtype, None)?;
    }
    Ok(())
}

fn execute_model(
    dtype: &str,
    normalization: Option<models::semantic::QkNormalization>,
) -> Result<()> {
    let root = std::env::temp_dir().join(format!(
        "libmir-metal-dense-model-{}-{dtype}-{normalization:?}",
        std::process::id()
    ));
    fs::create_dir_all(&root)?;
    write_config(&root)?;
    write_weights(&root.join("model.safetensors"), dtype)?;

    let layout = ModelLayout::inspect(&root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let mut contract = DecoderExecutionContract::discover(&layout, &decoder, &catalog)?;
    if let Some(normalization) = normalization {
        for layer in &mut contract.semantic.decoder.layers {
            if let models::semantic::MixerSpec::SoftmaxAttention(attention) = &mut layer.mixer {
                attention.qk_normalization = normalization;
            }
        }
    }
    let lowering = lowering::plan(&contract.semantic)?;
    let stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &stream)?;
    let model = DenseSwiGluModel::load(
        &tensors,
        &decoder,
        &contract.bindings,
        lowering.layers(),
        16,
        &stream,
    )?;
    let normalized = normalization != Some(models::semantic::QkNormalization::None);
    assert_eq!(model.layers[0].weights.attention.query_norm.is_some(), normalized);
    assert_eq!(model.layers[0].weights.attention.key_norm.is_some(), normalized);
    let mut cache = model.new_cache(&stream)?;
    let logits = model.forward_decode(&Array::from_u32(&[1], &[1, 1])?, &mut cache, 0, &stream)?;

    assert_eq!(model.layer_count(), 1);
    assert_eq!(logits.shape()?, vec![1, 1, 4]);
    assert!(logits.to_vec_f32(&stream)?.iter().all(|value| value.is_finite()));
    cache.reset()?;
    let state =
        model.forward_prefill_state(&Array::from_u32(&[1, 2], &[1, 2])?, &mut cache, 0, &stream)?;
    assert_eq!(state.shape()?, vec![1, 2, 2]);
    state.async_eval(&stream)?;
    stream.synchronize()?;
    assert_eq!(cache.cached_tokens()?, 2);
    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

fn write_config(root: &Path) -> Result<()> {
    let config = json!({
        "architectures": ["MistralForCausalLM"],
        "model_type": "mistral",
        "hidden_size": 2,
        "intermediate_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 4,
        "max_position_embeddings": 32,
        "hidden_act": "silu",
        "rms_norm_eps": 0.00001,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false
    });
    fs::write(root.join("config.json"), serde_json::to_vec(&config)?)?;
    Ok(())
}

fn write_weights(path: &Path, dtype: &str) -> Result<()> {
    let specs = [
        ("model.embed_tokens.weight", vec![4, 2]),
        ("model.norm.weight", vec![2]),
        ("lm_head.weight", vec![4, 2]),
        ("model.layers.0.input_layernorm.weight", vec![2]),
        ("model.layers.0.self_attn.q_proj.weight", vec![4, 2]),
        ("model.layers.0.self_attn.q_norm.weight", vec![4]),
        ("model.layers.0.self_attn.k_norm.weight", vec![4]),
        ("model.layers.0.self_attn.k_proj.weight", vec![4, 2]),
        ("model.layers.0.self_attn.v_proj.weight", vec![4, 2]),
        ("model.layers.0.self_attn.o_proj.weight", vec![2, 4]),
        ("model.layers.0.post_attention_layernorm.weight", vec![2]),
        ("model.layers.0.mlp.gate_proj.weight", vec![4, 2]),
        ("model.layers.0.mlp.up_proj.weight", vec![4, 2]),
        ("model.layers.0.mlp.down_proj.weight", vec![2, 4]),
    ];
    let mut header = Map::new();
    let mut offset = 0_usize;
    let mut payload = Vec::new();
    let element_bytes = if dtype == "F32" {
        4
    } else {
        2
    };
    for (name, shape) in &specs {
        let elements = shape.iter().product::<usize>();
        let bytes =
            elements.checked_mul(element_bytes).ok_or(crate::engine::Error::ShapeOverflow)?;
        header.insert(
            (*name).into(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes]}),
        );
        for _ in 0..elements {
            match dtype {
                "F32" => payload.extend_from_slice(&0.125_f32.to_le_bytes()),
                "F16" => payload.extend_from_slice(&0x3000_u16.to_le_bytes()),
                "BF16" => payload.extend_from_slice(&0x3e00_u16.to_le_bytes()),
                _ => return Err(crate::engine::Error::InvalidModel(dtype.into())),
            }
        }
        offset += bytes;
    }
    let mut header = serde_json::to_string(&Value::Object(header))?;
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut bytes = u64::try_from(header.len())?.to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&payload);
    fs::write(path, bytes)?;
    Ok(())
}

#[test]
fn honors_explicit_dense_normalization_even_when_checkpoint_contains_norm_weights() -> Result<()> {
    use models::semantic::QkNormalization;
    for normalization in [QkNormalization::None, QkNormalization::QueryKeyRms] {
        execute_model("F32", Some(normalization))?;
    }
    Ok(())
}
