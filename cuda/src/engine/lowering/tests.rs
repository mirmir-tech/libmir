use std::path::PathBuf;

use models::{
    execution::TaskExecutionPlan,
    layout::DecoderConfig,
    semantic::SemanticModelSpec,
    weights::{TensorCatalog, TensorInfo},
};
use serde_json::json;

use super::*;
use crate::Result;

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
    let mut catalog = TensorCatalog {
        tensors: [
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.k_norm.weight",
        ]
        .into_iter()
        .map(tensor)
        .collect(),
    };
    catalog.tensors.sort_by(|left, right| left.name.cmp(&right.name));
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;
    let plan = CudaDecoderPlan::lower(&spec, None)?;

    assert_eq!(
        plan.layers()[0].mixer.normalization(),
        Some(models::semantic::QkNormalization::QueryKeyRms)
    );
    assert_eq!(
        crate::admit_architecture(&TaskExecutionPlan::Generation { decoder }, Some(&spec))?
            .runtime(),
        Some(crate::CudaDecoderRuntime::Dense)
    );
    Ok(())
}

fn tensor(name: &str) -> TensorInfo {
    TensorInfo {
        name: name.into(),
        file: PathBuf::new(),
        container: models::layout::WeightContainer::SafeTensors,
        dtype: models::weights::DenseDType::Bf16.into(),
        // Per-head Q/K norms span one head; a query norm across the whole
        // projection would declare OLMo 2's norms across heads.
        shape: if name.ends_with("q_norm.weight") || name.ends_with("k_norm.weight") {
            vec![4]
        } else {
            vec![8]
        },
        data_start: 0,
        data_offsets: [0, 0],
    }
}
