use std::path::PathBuf;

use serde_json::json;

use super::{DecisionTensor, DecisionTensorPlan, EncoderTensor, HeadTensor, ScorerTensor};
use crate::{
    Result,
    layout::ModernBertConfig,
    weights::{TensorCatalog, TensorInfo},
};

fn encoder() -> Result<ModernBertConfig> {
    ModernBertConfig::from_json(
        &json!({
            "model_type": "modernbert", "hidden_size": 4, "intermediate_size": 3,
            "num_hidden_layers": 2, "num_attention_heads": 1, "vocab_size": 8,
            "max_position_embeddings": 16, "norm_eps": 1e-5, "hidden_activation": "gelu",
            "local_attention": 4, "global_attn_every_n_layers": 2,
            "global_rope_theta": 160_000.0, "local_rope_theta": 10_000.0,
            "attention_bias": false, "mlp_bias": false, "norm_bias": false
        })
        .to_string(),
    )
}

fn info(name: &str, shape: Vec<usize>) -> TensorInfo {
    TensorInfo {
        name: name.into(),
        file: PathBuf::from("model.safetensors"),
        dtype: "F16".into(),
        shape,
        data_start: 0,
        data_offsets: [0, 0],
    }
}

fn catalog(skip: &str, reshape: Option<(&str, Vec<usize>)>) -> TensorCatalog {
    let mut tensors = vec![
        info("encoder.embeddings.tok_embeddings.weight", vec![8, 4]),
        info("encoder.embeddings.norm.weight", vec![4]),
        info("encoder.final_norm.weight", vec![4]),
        info("type_emb.weight", vec![3, 4]),
        info("act_head.0.weight", vec![256, 8]),
    ];
    for layer in 0..2 {
        let prefix = format!("encoder.layers.{layer}");
        if layer > 0 {
            tensors.push(info(&format!("{prefix}.attn_norm.weight"), vec![4]));
        }
        tensors.extend([
            info(&format!("{prefix}.attn.Wqkv.weight"), vec![12, 4]),
            info(&format!("{prefix}.attn.Wo.weight"), vec![4, 4]),
            info(&format!("{prefix}.mlp_norm.weight"), vec![4]),
            info(&format!("{prefix}.mlp.Wi.weight"), vec![6, 4]),
            info(&format!("{prefix}.mlp.Wo.weight"), vec![4, 3]),
        ]);
    }
    for (suffix, shape) in [
        ("norm1.weight", vec![4]),
        ("norm1.bias", vec![4]),
        ("self_attn.in_proj_weight", vec![12, 4]),
        ("self_attn.in_proj_bias", vec![12]),
        ("self_attn.out_proj.weight", vec![4, 4]),
        ("self_attn.out_proj.bias", vec![4]),
        ("norm2.weight", vec![4]),
        ("norm2.bias", vec![4]),
        ("linear1.weight", vec![16, 4]),
        ("linear1.bias", vec![16]),
        ("linear2.weight", vec![4, 16]),
        ("linear2.bias", vec![4]),
    ] {
        tensors.push(info(&format!("head.layers.0.{suffix}"), shape));
    }
    for (suffix, shape) in [
        ("0.weight", vec![4]),
        ("0.bias", vec![4]),
        ("1.weight", vec![4, 4]),
        ("1.bias", vec![4]),
        ("3.weight", vec![1, 4]),
        ("3.bias", vec![1]),
    ] {
        tensors.push(info(&format!("scorer.{suffix}"), shape));
    }
    tensors.retain(|tensor| tensor.name != skip);
    if let Some((name, shape)) = reshape {
        tensors
            .iter_mut()
            .filter(|tensor| tensor.name == name)
            .for_each(|tensor| tensor.shape = shape.clone());
    }
    TensorCatalog::new(tensors)
}

#[test]
fn binds_every_inference_tensor_and_ignores_the_act_head() -> Result<()> {
    let plan = DecisionTensorPlan::discover(&encoder()?, 1, &catalog("", None))?;
    assert_eq!(plan.iter().count(), 4 + 5 + 6 + 12 + 6);
    let layer = |tensor| DecisionTensor::Encoder { layer: 0, tensor };
    assert!(plan.get(layer(EncoderTensor::AttentionNorm)).is_err());
    assert_eq!(plan.get(layer(EncoderTensor::MlpInput))?.shape, vec![6, 4]);
    let head = DecisionTensor::Head { layer: 0, tensor: HeadTensor::UpWeight };
    assert_eq!(plan.get(head)?.name, "head.layers.0.linear1.weight");
    assert_eq!(plan.get(DecisionTensor::Scorer(ScorerTensor::LogitBias))?.shape, vec![1]);
    Ok(())
}

#[test]
fn rejects_missing_and_misshapen_tensors() -> Result<()> {
    let encoder = encoder()?;
    assert!(DecisionTensorPlan::discover(&encoder, 1, &catalog("scorer.3.bias", None)).is_err());
    let misshapen = catalog("", Some(("encoder.layers.1.mlp.Wi.weight", vec![3, 4])));
    assert!(DecisionTensorPlan::discover(&encoder, 1, &misshapen).is_err());
    assert!(DecisionTensorPlan::discover(&encoder, 2, &catalog("", None)).is_err());
    Ok(())
}
