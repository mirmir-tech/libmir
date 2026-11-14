//! Vision checks: a pipeline without a tower on the backend leaves the
//! decoder admitted text-only.

use foundation::model::BackendTarget;
use models::{
    layout::{ImageProcessorConfig, VisionConfig, VisionPipeline},
    weights::TensorReadiness,
};
use serde_json::json;

use super::super::{AdmissionCheck, AdmissionCheckKind, AdmissionStatus, aggregate, vision_check};

#[test]
fn metal_admits_a_pixel_shuffle_checkpoint_text_only() -> models::Result<()> {
    let vision = VisionConfig::from_value(&json!({
        "downsample_ratio": 0.5,
        "ps_version": "v2",
        "vit_hidden_size": 64,
        "projector_hidden_size": 256,
        "img_context_token_id": 18,
        "image_tag_type": "internvl",
        "llm_config": {"hidden_size": 128},
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
    }))?
    .ok_or_else(|| models::ModelsError::InvalidConfig("vision not discovered".into()))?;
    let processor = ImageProcessorConfig::from_value(
        &json!({
            "downsample_ratio": 0.5,
            "max_num_patches": 1024,
            "min_num_patches": 16,
            "norm_mean": [0.5, 0.5, 0.5],
            "norm_std": [0.5, 0.5, 0.5],
            "patch_size": 16
        }),
        VisionPipeline::PixelShuffleEncoder,
    )?;
    let readiness = TensorReadiness {
        required: 1,
        present: 1,
        missing: Vec::new(),
        dtypes: vec!["BF16".into()],
    };
    let check = |backend| vision_check(&vision, Some(&readiness), Some(&processor), &backend);
    let metal = check(BackendTarget::Metal);
    assert_eq!(
        (metal.kind, metal.status),
        (AdmissionCheckKind::Vision, AdmissionStatus::Partial)
    );
    assert_eq!(check(BackendTarget::Cuda).status, AdmissionStatus::Supported);
    let decoder = AdmissionCheck {
        kind: AdmissionCheckKind::Architecture,
        status: AdmissionStatus::Supported,
        detail: String::new(),
    };
    assert_eq!(aggregate(&[decoder, metal]), AdmissionStatus::Partial);
    Ok(())
}
