use super::*;
use crate::EncodingKind;

mod gguf;
mod mlx;

fn config() -> serde_json::Value {
    json!({
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "num_local_experts": 2,
        "num_experts_per_tok": 1,
        "hidden_act": "silu",
        "attention_bias": true,
        "swiglu_limit": 7.0,
        "layer_types": ["full_attention"],
        "rope_theta": 150_000.0,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 4.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "original_max_position_embeddings": 32
        },
        "tie_word_embeddings": false
    })
}

#[test]
fn admits_dense_clamped_routed_generation_on_metal_and_cuda() -> Result<()> {
    let catalog = catalog("F32", "F32", "remote.safetensors");
    let contract = RemoteModelContract::inspect_generation(&config(), &catalog)?;
    let metal = contract.admission(BackendTarget::Metal);
    let cuda = contract.admission(BackendTarget::Cuda);

    for report in [&metal, &cuda] {
        assert_eq!(report.status, AdmissionStatus::Supported);
        assert!(report.checks.iter().any(|check| {
            check.kind() == AdmissionCheckKind::Encoding(EncodingKind::Dense)
                && check.status() == AdmissionStatus::Supported
        }));
    }
    Ok(())
}

/// GPT-OSS from an F16 or BF16 GGUF: half-precision projections beside the
/// F32 biases GGUF keeps, which its loaders convert with the weights.
#[test]
fn admits_half_precision_gguf_projections_beside_f32_biases() -> Result<()> {
    for dtype in ["F16", "BF16"] {
        let catalog = catalog(dtype, "F32", "gpt-oss-20b.gguf");
        let contract = RemoteModelContract::inspect_generation(&config(), &catalog)?;
        for backend in [BackendTarget::Metal, BackendTarget::Cuda] {
            let report = contract.admission(backend);
            assert_eq!(report.status, AdmissionStatus::Supported, "{dtype} {report:?}");
        }
    }
    Ok(())
}

/// `SafeTensors` biases load as stored, so an F32 bias beside F16 weights
/// would widen the projection's output: the contract refuses it.
#[test]
fn refuses_f32_safetensors_biases_beside_half_precision_projections() {
    let catalog = catalog("F16", "F32", "model.safetensors");
    let error = RemoteModelContract::inspect_generation(&config(), &catalog).err();
    assert!(
        error
            .as_ref()
            .is_some_and(|error| error.to_string().contains("expected dtype F16, found F32")),
        "{error:?}"
    );
}

/// The tensors of a one-layer dense GPT-OSS: `projection` weights, `bias`
/// biases, F32 norms and sinks, all in `file`.
fn catalog(projection: &str, bias: &str, file: &str) -> TensorCatalog {
    let layer = "model.layers.0";
    let specs = [
        ("model.embed_tokens.weight".into(), vec![64, 32]),
        ("model.norm.weight".into(), vec![32]),
        ("lm_head.weight".into(), vec![64, 32]),
        (format!("{layer}.input_layernorm.weight"), vec![32]),
        (format!("{layer}.self_attn.q_proj.weight"), vec![32, 32]),
        (format!("{layer}.self_attn.q_proj.bias"), vec![32]),
        (format!("{layer}.self_attn.k_proj.weight"), vec![16, 32]),
        (format!("{layer}.self_attn.k_proj.bias"), vec![16]),
        (format!("{layer}.self_attn.v_proj.weight"), vec![16, 32]),
        (format!("{layer}.self_attn.v_proj.bias"), vec![16]),
        (format!("{layer}.self_attn.o_proj.weight"), vec![32, 32]),
        (format!("{layer}.self_attn.o_proj.bias"), vec![32]),
        (format!("{layer}.self_attn.sinks"), vec![4]),
        (format!("{layer}.post_attention_layernorm.weight"), vec![32]),
        (format!("{layer}.mlp.router.weight"), vec![2, 32]),
        (format!("{layer}.mlp.experts.gate_proj.weight"), vec![2, 32, 32]),
        (format!("{layer}.mlp.experts.up_proj.weight"), vec![2, 32, 32]),
        (format!("{layer}.mlp.experts.down_proj.weight"), vec![2, 32, 32]),
    ];
    TensorCatalog::new(
        specs
            .into_iter()
            .map(|(name, shape)| {
                // Norms and sinks are vectors, like the biases.
                let dtype = if name.ends_with("_proj.bias") {
                    bias
                } else if shape.len() == 1 {
                    "F32"
                } else {
                    projection
                };
                let mut tensor = tensor_with_dtype(&name, shape, dtype);
                tensor.file = file.into();
                tensor
            })
            .collect(),
    )
}
