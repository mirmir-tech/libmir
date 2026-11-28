use std::path::PathBuf;

use models::{
    layout::VisionConfig,
    weights::{PredictionTensorRole, TensorBinding, TensorCatalog, TensorInfo, WeightBindingPlan},
};
use serde_json::json;

use super::super::*;

#[test]
fn a_vision_load_still_leaves_prediction_modules_on_disk() -> crate::Result<()> {
    let binding = |source: &str, role| TensorBinding {
        role,
        source: source.into(),
        shape: vec![8],
        logical_shape: None,
        transforms: Vec::new(),
        storage: TensorStorage::Dense {
            dtype: models::weights::DenseDType::Bf16,
            bias: None,
        },
    };
    let prediction = LogicalTensorRole::Prediction {
        depth: 0,
        tensor: PredictionTensorRole::EmbeddingNorm,
    };
    let auxiliary = LogicalTensorRole::Auxiliary { path: "vision".into() };
    let bindings = WeightBindingPlan {
        tensors: vec![binding("mtp.enorm.weight", prediction), binding("vision.weight", auxiliary)],
    };
    let tensor = |name: &str, bytes: u64| TensorInfo {
        name: name.into(),
        file: PathBuf::from("model.safetensors"),
        container: models::layout::WeightContainer::SafeTensors,
        dtype: models::weights::DenseDType::Bf16.into(),
        shape: vec![8],
        data_start: 0,
        data_offsets: [0, bytes],
    };
    let catalog = TensorCatalog {
        tensors: vec![tensor("mtp.enorm.weight", 16), tensor("vision.weight", 64)],
    };
    let vision = VisionConfig::from_value(&pixel_shuffle_config())?
        .ok_or_else(|| models::ModelsError::InvalidConfig("vision not discovered".into()))?;
    let counts = |unloaded: Option<UnloadedTensors>| {
        unloaded.map(|unloaded| (unloaded.tensors, unloaded.bytes, unloaded.prediction_bytes))
    };
    assert_eq!(counts(unloaded(&catalog, Some(&bindings), None)), Some((2, 80, 16)));
    // The tower loads alongside the decoder; the prediction module does not.
    assert_eq!(counts(unloaded(&catalog, Some(&bindings), Some(&vision))), Some((1, 16, 16)));
    Ok(())
}

fn pixel_shuffle_config() -> serde_json::Value {
    json!({
        "downsample_ratio": 0.5,
        "ps_version": "v2",
        "vit_hidden_size": 64,
        "projector_hidden_size": 256,
        "img_context_token_id": 18,
        "image_tag_type": "internvl",
        "llm_config": {"hidden_size": 128, "num_nextn_predict_layers": 1},
        "vision_config": {
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "patch_size": 16,
            "mlp_ratio": 4.0,
            "num_cls_tokens": 1,
            "num_registers": 0,
            "max_img_size": 256,
            "layer_norm_eps": 1e-06,
            "hidden_act": "gelu"
        }
    })
}
