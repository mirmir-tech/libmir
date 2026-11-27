//! Fused gate/up projections and multi-layer BF16 generation for GPTQ (v1, v2)
//! and AWQ checkpoints whose scales are F16 while model activations stay BF16.

use std::{fs, path::Path};

use models::{
    execution::DecoderExecutionContract,
    layout::{DecoderConfig, ModelLayout},
    weights::TensorCatalog,
};
use serde_json::{Map, Value, json};

use super::super::super::DenseSwiGluModel;
use crate::engine::{Array, Dtype, ModelTensors, Result, Stream, lowering};

const HIDDEN: usize = 128;
const INTERMEDIATE: usize = 128;
const HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const GROUP: usize = 128;
const VOCAB: usize = 16;

#[derive(Clone, Copy, Debug)]
enum Format {
    GptqV1,
    GptqV2,
    Awq,
}

#[test]
fn bf16_generation_keeps_kv_cache_dtype_for_gptq_and_awq() -> Result<()> {
    for format in [Format::GptqV1, Format::GptqV2, Format::Awq] {
        run_generation(format)?;
    }
    Ok(())
}

fn run_generation(format: Format) -> Result<()> {
    let root = std::env::temp_dir()
        .join(format!("libmir-metal-fused-dtype-{}-{format:?}", std::process::id()));
    fs::create_dir_all(&root)?;
    write_config(&root, format)?;
    write_weights(&root.join("model.safetensors"), format)?;

    let layout = ModelLayout::inspect(&root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let contract = DecoderExecutionContract::discover(&layout, &decoder, &catalog)?;
    let lowering = lowering::plan(&decoder, &contract.semantic)?;
    let tensors = ModelTensors::load(&root, &Stream::new_cpu()?)?;
    let stream = Stream::new_gpu()?;
    let model = DenseSwiGluModel::load(
        &tensors,
        &decoder,
        &contract.semantic.decoder,
        &contract.bindings,
        lowering.layers(),
        lowering.final_norm(),
        16,
        &stream,
    )?;
    assert_eq!(model.layer_count(), 2);
    assert!(model.layers[0].fused_gate_up.is_some(), "fused gate/up must run on decode");

    let mut cache = model.new_cache(&stream)?;
    let prefill =
        model.forward_prefill(&Array::from_u32(&[1, 2], &[1, 2])?, &mut cache, 0, &stream)?;
    assert_eq!(prefill.dtype()?, Dtype::Bfloat16);
    assert!(prefill.to_vec_f32(&stream)?.iter().all(|value| value.is_finite()));

    for position in [2_i32, 3_i32] {
        let decode = model.forward_decode(
            &Array::from_u32(&[3], &[1, 1])?,
            &mut cache,
            position,
            &stream,
        )?;
        assert_eq!(decode.dtype()?, Dtype::Bfloat16);
        assert!(decode.to_vec_f32(&stream)?.iter().all(|value| value.is_finite()));
    }
    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

fn write_config(root: &Path, format: Format) -> Result<()> {
    let quantization = match format {
        Format::GptqV1 => json!({
            "quant_method": "gptq", "bits": 4, "group_size": GROUP,
            "desc_act": false, "sym": true, "checkpoint_format": "gptq",
        }),
        Format::GptqV2 => json!({
            "quant_method": "gptq", "bits": 4, "group_size": GROUP,
            "desc_act": false, "sym": true, "checkpoint_format": "gptq_v2",
        }),
        Format::Awq => json!({
            "quant_method": "awq", "bits": 4, "group_size": GROUP,
            "version": "gemm", "zero_point": true,
        }),
    };
    let config = json!({
        "model_type": "qwen3", "hidden_size": HIDDEN, "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": 2, "num_attention_heads": HEADS, "num_key_value_heads": HEADS,
        "head_dim": HEAD_DIM, "vocab_size": VOCAB, "rms_norm_eps": 1e-6,
        "tie_word_embeddings": false, "torch_dtype": "bfloat16",
        "quantization_config": quantization,
    });
    fs::write(root.join("config.json"), serde_json::to_vec(&config)?)?;
    Ok(())
}

fn push_bf16(
    header: &mut Map<String, Value>,
    payload: &mut Vec<u8>,
    name: String,
    shape: &[usize],
) {
    let start = payload.len();
    for _ in 0..shape.iter().product::<usize>() {
        payload.extend_from_slice(&0x3F80_u16.to_le_bytes());
    }
    let end = payload.len();
    header.insert(name, json!({"dtype": "BF16", "shape": shape, "data_offsets": [start, end]}));
}

fn push_i32(
    header: &mut Map<String, Value>,
    payload: &mut Vec<u8>,
    name: String,
    shape: &[usize],
    word: i32,
) {
    let start = payload.len();
    for _ in 0..shape.iter().product::<usize>() {
        payload.extend_from_slice(&word.to_le_bytes());
    }
    let end = payload.len();
    header.insert(name, json!({"dtype": "I32", "shape": shape, "data_offsets": [start, end]}));
}

fn write_weights(path: &Path, format: Format) -> Result<()> {
    let mut header = Map::new();
    let mut payload = Vec::new();
    push_bf16(&mut header, &mut payload, "model.embed_tokens.weight".into(), &[VOCAB, HIDDEN]);
    push_bf16(&mut header, &mut payload, "model.norm.weight".into(), &[HIDDEN]);
    push_bf16(&mut header, &mut payload, "lm_head.weight".into(), &[VOCAB, HIDDEN]);
    for layer in 0..2 {
        let prefix = format!("model.layers.{layer}");
        push_bf16(&mut header, &mut payload, format!("{prefix}.input_layernorm.weight"), &[HIDDEN]);
        push_bf16(
            &mut header,
            &mut payload,
            format!("{prefix}.post_attention_layernorm.weight"),
            &[HIDDEN],
        );
        push_bf16(
            &mut header,
            &mut payload,
            format!("{prefix}.self_attn.q_norm.weight"),
            &[HEAD_DIM],
        );
        push_bf16(
            &mut header,
            &mut payload,
            format!("{prefix}.self_attn.k_norm.weight"),
            &[HEAD_DIM],
        );
        for name in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            push_quantized(&mut header, &mut payload, &format!("{prefix}.{name}"), format);
        }
    }
    let mut header_bytes = serde_json::to_vec(&Value::Object(header))?;
    while !header_bytes.len().is_multiple_of(8) {
        header_bytes.push(b' ');
    }
    let mut data = u64::try_from(header_bytes.len())?.to_le_bytes().to_vec();
    data.extend_from_slice(&header_bytes);
    data.extend_from_slice(&payload);
    fs::write(path, data)?;
    Ok(())
}

fn push_quantized(
    header: &mut Map<String, Value>,
    payload: &mut Vec<u8>,
    prefix: &str,
    format: Format,
) {
    let (qweight_shape, qzeros_shape) = match format {
        Format::GptqV1 | Format::GptqV2 => ([HIDDEN / 8, HIDDEN], [HIDDEN / GROUP, HIDDEN / 8]),
        Format::Awq => ([HIDDEN, HIDDEN / 8], [HIDDEN / GROUP, HIDDEN / 8]),
    };
    push_i32(header, payload, format!("{prefix}.qweight"), &qweight_shape, 0x3333_3333_i32);
    push_i32(header, payload, format!("{prefix}.qzeros"), &qzeros_shape, 0x2222_2222_i32);
    let start = payload.len();
    for _ in 0..(HIDDEN / GROUP) * HIDDEN {
        payload.extend_from_slice(&0x3400_u16.to_le_bytes());
    }
    let end = payload.len();
    header.insert(
        format!("{prefix}.scales"),
        json!({"dtype": "F16", "shape": [HIDDEN / GROUP, HIDDEN], "data_offsets": [start, end]}),
    );
    if matches!(format, Format::GptqV1 | Format::GptqV2) {
        push_i32(header, payload, format!("{prefix}.g_idx"), &[HIDDEN], 0);
    }
}
