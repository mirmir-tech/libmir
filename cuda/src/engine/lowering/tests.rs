use std::path::PathBuf;

use models::{
    layout::DecoderConfig,
    semantic::SemanticModelSpec,
    weights::{TensorCatalog, TensorInfo},
};
use serde_json::json;

use super::*;

#[test]
fn lowers_dense_qk_normalization_from_semantics() -> Result<()> {
    let decoder = DecoderConfig::from_value(&json!({
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 32,
        "hidden_act": "silu",
        "model_type": "misleading"
    }))?;
    let catalog = TensorCatalog {
        tensors: [
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.k_norm.weight",
        ]
        .into_iter()
        .map(tensor)
        .collect(),
    };
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;
    let plan = CudaDecoderPlan::lower(&spec);

    assert!(plan.all_dense());
    assert_eq!(plan.graph_normalization()?, KernelNormalization::QUERY_KEY);
    Ok(())
}

fn tensor(name: &str) -> TensorInfo {
    TensorInfo {
        name: name.into(),
        file: PathBuf::new(),
        dtype: "BF16".into(),
        shape: vec![8],
        data_start: 0,
        data_offsets: [0, 0],
    }
}
