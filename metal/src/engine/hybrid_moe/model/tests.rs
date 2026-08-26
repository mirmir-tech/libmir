use std::{fs, path::Path};

use models::{
    execution::DecoderExecutionContract,
    layout::{DecoderConfig, ModelLayout},
    weights::TensorCatalog,
};
use serde_json::{Map, Value, json};

use super::HybridMoeModel;
use crate::engine::{Array, ModelTensors, Result, Stream, lowering};

#[test]
fn executes_a_complete_dense_hybrid_moe_model() -> Result<()> {
    for fused_experts in [false, true] {
        execute_model(fused_experts)?;
    }
    Ok(())
}

fn execute_model(fused_experts: bool) -> Result<()> {
    let root = std::env::temp_dir()
        .join(format!("libmir-metal-dense-hybrid-moe-{}-{fused_experts}", std::process::id()));
    fs::create_dir_all(&root)?;
    write_config(&root)?;
    write_weights(&root.join("model.safetensors"), fused_experts)?;

    let layout = ModelLayout::inspect(&root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let contract = DecoderExecutionContract::discover(&layout, &decoder, &catalog)?;
    let lowering = lowering::plan(&contract.semantic)?;
    let stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &stream)?;
    let model = HybridMoeModel::load_bindings(
        &tensors,
        &decoder,
        &contract.bindings,
        lowering.layers(),
        16,
        &stream,
    )?;
    let mut cache = model.new_cache(&stream)?;
    let logits = model.forward_decode(&Array::from_u32(&[1], &[1, 1])?, &mut cache, 0, &stream)?;

    assert_eq!(model.layer_count(), 1);
    assert_eq!(logits.shape()?, vec![1, 1, 8]);
    assert!(logits.to_vec_f32(&stream)?.iter().all(|value| value.is_finite()));
    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

fn write_config(root: &Path) -> Result<()> {
    let config = json!({
        "architectures": ["HybridForCausalLM"],
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 8,
        "max_position_embeddings": 32,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 4,
        "hidden_act": "gelu_pytorch_tanh",
        "attention_k_eq_v": true,
        "rms_norm_eps": 0.000_001,
        "tie_word_embeddings": true
    });
    fs::write(root.join("config.json"), serde_json::to_vec(&config)?)?;
    Ok(())
}

fn write_weights(path: &Path, fused_experts: bool) -> Result<()> {
    let layer = "language_model.model.layers.0";
    let mut specs = vec![
        ("language_model.model.embed_tokens.weight".into(), vec![8, 4]),
        ("language_model.model.norm.weight".into(), vec![4]),
        (format!("{layer}.input_layernorm.weight"), vec![4]),
        (format!("{layer}.self_attn.q_proj.weight"), vec![4, 4]),
        (format!("{layer}.self_attn.k_proj.weight"), vec![4, 4]),
        (format!("{layer}.self_attn.o_proj.weight"), vec![4, 4]),
        (format!("{layer}.self_attn.q_norm.weight"), vec![4]),
        (format!("{layer}.self_attn.k_norm.weight"), vec![4]),
        (format!("{layer}.post_attention_layernorm.weight"), vec![4]),
        (format!("{layer}.pre_feedforward_layernorm.weight"), vec![4]),
        (format!("{layer}.mlp.gate_proj.weight"), vec![8, 4]),
        (format!("{layer}.mlp.up_proj.weight"), vec![8, 4]),
        (format!("{layer}.mlp.down_proj.weight"), vec![4, 8]),
        (format!("{layer}.post_feedforward_layernorm_1.weight"), vec![4]),
        (format!("{layer}.router.proj.weight"), vec![2, 4]),
        (format!("{layer}.router.scale"), vec![4]),
        (format!("{layer}.router.per_expert_scale"), vec![2]),
        (format!("{layer}.pre_feedforward_layernorm_2.weight"), vec![4]),
        (format!("{layer}.post_feedforward_layernorm_2.weight"), vec![4]),
        (format!("{layer}.post_feedforward_layernorm.weight"), vec![4]),
        (format!("{layer}.layer_scalar"), vec![1]),
    ];
    if fused_experts {
        specs.extend([
            (format!("{layer}.experts.gate_up_proj"), vec![2, 8, 4]),
            (format!("{layer}.experts.down_proj"), vec![2, 4, 4]),
        ]);
    } else {
        specs.extend([
            (format!("{layer}.experts.switch_glu.gate_proj.weight"), vec![2, 4, 4]),
            (format!("{layer}.experts.switch_glu.up_proj.weight"), vec![2, 4, 4]),
            (format!("{layer}.experts.switch_glu.down_proj.weight"), vec![2, 4, 4]),
        ]);
    }
    let mut header = Map::new();
    let mut offset = 0_usize;
    let mut payload = Vec::new();
    for (name, shape) in &specs {
        let elements = shape.iter().product::<usize>();
        let bytes = elements.checked_mul(4).ok_or(crate::engine::Error::ShapeOverflow)?;
        header.insert(
            name.clone(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + bytes]}),
        );
        for _ in 0..elements {
            payload.extend_from_slice(&0.125_f32.to_le_bytes());
        }
        offset += bytes;
    }
    write_safetensors(path, header, &payload)
}

fn write_safetensors(path: &Path, header: Map<String, Value>, payload: &[u8]) -> Result<()> {
    let mut header = serde_json::to_string(&Value::Object(header))?;
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut bytes = u64::try_from(header.len())?.to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(payload);
    fs::write(path, bytes)?;
    Ok(())
}
