//! Qwen2's biased attention projections: CUDA's dense decoder adds biases
//! only to GGML layers, so an F16 GGUF or BF16 `SafeTensors` checkpoint it
//! would refuse at load is refused at admission; Metal adds them in any
//! dense storage.

use super::*;

fn config() -> serde_json::Value {
    json!({
        "model_type": "qwen2",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 64,
        "hidden_act": "silu"
    })
}

/// Qwen2 with `projection` weights and `bias` biases in `file`.
fn catalog(projection: &str, bias: &str, file: &str) -> TensorCatalog {
    let layer = "model.layers.0";
    let mut tensors = vec![
        tensor_with_dtype("model.norm.weight", vec![32], "F32"),
        tensor_with_dtype(&format!("{layer}.input_layernorm.weight"), vec![32], "F32"),
        tensor_with_dtype(&format!("{layer}.post_attention_layernorm.weight"), vec![32], "F32"),
    ];
    let matrices = [
        ("model.embed_tokens.weight", [64, 32]),
        ("lm_head.weight", [64, 32]),
        ("model.layers.0.self_attn.q_proj.weight", [32, 32]),
        ("model.layers.0.self_attn.k_proj.weight", [16, 32]),
        ("model.layers.0.self_attn.v_proj.weight", [16, 32]),
        ("model.layers.0.self_attn.o_proj.weight", [32, 32]),
        ("model.layers.0.mlp.gate_proj.weight", [64, 32]),
        ("model.layers.0.mlp.up_proj.weight", [64, 32]),
        ("model.layers.0.mlp.down_proj.weight", [32, 64]),
    ];
    for (name, [rows, columns]) in matrices {
        let mut tensor = tensor_with_dtype(name, vec![rows, columns], projection);
        // A Q8_0 block holds 32 values in 34 bytes.
        if projection == "Q8_0" {
            tensor.data_offsets = [0, u64::try_from(rows * columns / 32 * 34).unwrap_or(0)];
        }
        tensors.push(tensor);
    }
    for (projection, rows) in [("q", 32), ("k", 16), ("v", 16)] {
        let name = format!("{layer}.self_attn.{projection}_proj.bias");
        tensors.push(tensor_with_dtype(&name, vec![rows], bias));
    }
    for tensor in &mut tensors {
        tensor.file = file.into();
    }
    TensorCatalog::new(tensors)
}

fn admission(catalog: &TensorCatalog) -> Result<(AdmissionStatus, BackendAdmissionReport)> {
    let contract = RemoteModelContract::inspect_generation(&config(), catalog)?;
    Ok((
        contract.admission(BackendTarget::Metal).status,
        contract.admission(BackendTarget::Cuda),
    ))
}

#[test]
fn cuda_refuses_biased_projections_outside_ggml_layers() -> Result<()> {
    for catalog in
        [catalog("F16", "F32", "qwen2.gguf"), catalog("BF16", "BF16", "qwen2.safetensors")]
    {
        let (metal, cuda) = admission(&catalog)?;
        assert_eq!(metal, AdmissionStatus::Supported);
        assert_eq!(cuda.status, AdmissionStatus::Unsupported, "{cuda:?}");
        assert!(
            cuda.checks.iter().any(|check| {
                check.kind() == AdmissionCheckKind::Encoding(EncodingKind::Dense)
                    && check.status() == AdmissionStatus::Unsupported
                    && check.detail.contains("only to GGML dense layers")
                    && check.detail.contains("q_proj.weight")
            }),
            "{cuda:?}"
        );
    }
    Ok(())
}

#[test]
fn cuda_admits_biased_ggml_projections() -> Result<()> {
    let (metal, cuda) = admission(&catalog("Q8_0", "F32", "qwen2.gguf"))?;
    assert_ne!(metal, AdmissionStatus::Unsupported);
    assert!(
        !cuda
            .checks
            .iter()
            .any(|check| check.detail.contains("only to GGML dense layers")),
        "{cuda:?}"
    );
    assert_ne!(cuda.status, AdmissionStatus::Unsupported, "{cuda:?}");
    Ok(())
}
