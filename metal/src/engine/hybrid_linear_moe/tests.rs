use std::{fs, path::Path};

use models::{
    execution::DecoderExecutionContract,
    layout::{DecoderConfig, ModelLayout},
    weights::TensorCatalog,
};
use serde_json::{Map, Value, json};

use super::HybridLinearMoeModel;
use crate::engine::{Array, ModelTensors, Result, Stream, lowering};

#[test]
fn executes_a_complete_dense_shared_routed_model() -> Result<()> {
    let root = std::env::temp_dir()
        .join(format!("libmir-metal-dense-shared-routed-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    write_config(&root)?;
    write_weights(&root.join("model.safetensors"))?;

    let layout = ModelLayout::inspect(&root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let contract = DecoderExecutionContract::discover(&layout, &decoder, &catalog)?;
    let lowering = lowering::plan(&contract.semantic)?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &load_stream)?;
    let stream = Stream::new_gpu()?;
    let model = HybridLinearMoeModel::load(
        &tensors,
        &decoder,
        &contract.bindings,
        lowering.layers(),
        16,
        &stream,
    )?;
    let mut cache = model.new_cache(&stream)?;
    let logits = model.forward_decode(&Array::from_u32(&[1], &[1, 1])?, &mut cache, 0, &stream)?;

    assert_eq!(logits.shape()?, vec![1, 1, 64]);
    assert!(logits.to_vec_f32(&stream)?.iter().all(|value| value.is_finite()));
    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn packed_prefill_matches_independent_rows() -> Result<()> {
    let root = std::env::temp_dir()
        .join(format!("libmir-metal-packed-shared-routed-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    write_config(&root)?;
    write_weights(&root.join("model.safetensors"))?;

    let layout = ModelLayout::inspect(&root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let contract = DecoderExecutionContract::discover(&layout, &decoder, &catalog)?;
    let lowering = lowering::plan(&contract.semantic)?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &load_stream)?;
    let stream = Stream::new_gpu()?;
    let model = HybridLinearMoeModel::load(
        &tensors,
        &decoder,
        &contract.bindings,
        lowering.layers(),
        16,
        &stream,
    )?;
    let rows = [[1, 2, 3, 4], [5, 6, 7, 8]];
    let mut scalar_caches = [model.new_cache(&stream)?, model.new_cache(&stream)?];
    let scalar = rows
        .iter()
        .zip(&mut scalar_caches)
        .map(|(tokens, cache)| {
            model.forward_prefill(&Array::from_u32(tokens, &[1, 4])?, cache, 0, &stream)
        })
        .collect::<Result<Vec<_>>>()?;
    let scalar = Array::concatenate(&scalar.iter().collect::<Vec<_>>(), 0, &stream)?;

    let mut packed_caches = [model.new_cache(&stream)?, model.new_cache(&stream)?];
    let mut packed_cache_refs = packed_caches.iter_mut().collect::<Vec<_>>();
    let packed = model.forward_packed_prefill_state(
        &Array::from_u32(&rows.concat(), &[2, 4])?,
        &mut packed_cache_refs,
        &[0, 0],
        &stream,
    )?;
    let expected = scalar.to_vec_f32(&stream)?;
    let actual = packed.to_vec_f32(&stream)?;
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().zip(expected).all(|(actual, expected)| {
        (actual - expected).abs() <= 1.0e-4 * expected.abs().max(1.0)
    }));
    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

fn write_config(root: &Path) -> Result<()> {
    let config = json!({
        "architectures": ["HybridForCausalLM"],
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
        "attn_output_gate": true,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 1,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "rms_norm_eps": 0.000_001,
        "tie_word_embeddings": false
    });
    fs::write(root.join("config.json"), serde_json::to_vec(&config)?)?;
    Ok(())
}

fn write_weights(path: &Path) -> Result<()> {
    let layer = "language_model.model.layers.0";
    let full = "language_model.model.layers.1";
    let specs = [
        ("language_model.model.embed_tokens.weight".into(), vec![64, 32]),
        ("language_model.model.norm.weight".into(), vec![32]),
        ("language_model.lm_head.weight".into(), vec![64, 32]),
        (format!("{layer}.input_layernorm.weight"), vec![32]),
        (format!("{layer}.post_attention_layernorm.weight"), vec![32]),
        (format!("{layer}.mlp.gate.weight"), vec![8, 32]),
        (format!("{layer}.mlp.switch_mlp.gate_proj.weight"), vec![8, 16, 32]),
        (format!("{layer}.mlp.switch_mlp.up_proj.weight"), vec![8, 16, 32]),
        (format!("{layer}.mlp.switch_mlp.down_proj.weight"), vec![8, 32, 16]),
        (format!("{layer}.mlp.shared_expert.gate_proj.weight"), vec![16, 32]),
        (format!("{layer}.mlp.shared_expert.up_proj.weight"), vec![16, 32]),
        (format!("{layer}.mlp.shared_expert.down_proj.weight"), vec![32, 16]),
        (format!("{layer}.mlp.shared_expert_gate.weight"), vec![1, 32]),
        (format!("{layer}.linear_attn.in_proj_qkv.weight"), vec![96, 32]),
        (format!("{layer}.linear_attn.in_proj_z.weight"), vec![32, 32]),
        (format!("{layer}.linear_attn.in_proj_a.weight"), vec![1, 32]),
        (format!("{layer}.linear_attn.in_proj_b.weight"), vec![1, 32]),
        (format!("{layer}.linear_attn.out_proj.weight"), vec![32, 32]),
        (format!("{layer}.linear_attn.conv1d.weight"), vec![96, 4, 1]),
        (format!("{layer}.linear_attn.norm.weight"), vec![32]),
        (format!("{layer}.linear_attn.A_log"), vec![1]),
        (format!("{layer}.linear_attn.dt_bias"), vec![1]),
        (format!("{full}.input_layernorm.weight"), vec![32]),
        (format!("{full}.post_attention_layernorm.weight"), vec![32]),
        (format!("{full}.mlp.gate.weight"), vec![8, 32]),
        (format!("{full}.mlp.switch_mlp.gate_proj.weight"), vec![8, 16, 32]),
        (format!("{full}.mlp.switch_mlp.up_proj.weight"), vec![8, 16, 32]),
        (format!("{full}.mlp.switch_mlp.down_proj.weight"), vec![8, 32, 16]),
        (format!("{full}.mlp.shared_expert.gate_proj.weight"), vec![16, 32]),
        (format!("{full}.mlp.shared_expert.up_proj.weight"), vec![16, 32]),
        (format!("{full}.mlp.shared_expert.down_proj.weight"), vec![32, 16]),
        (format!("{full}.mlp.shared_expert_gate.weight"), vec![1, 32]),
        (format!("{full}.self_attn.q_proj.weight"), vec![64, 32]),
        (format!("{full}.self_attn.k_proj.weight"), vec![16, 32]),
        (format!("{full}.self_attn.v_proj.weight"), vec![16, 32]),
        (format!("{full}.self_attn.o_proj.weight"), vec![32, 32]),
        (format!("{full}.self_attn.q_norm.weight"), vec![8]),
        (format!("{full}.self_attn.k_norm.weight"), vec![8]),
    ];
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
