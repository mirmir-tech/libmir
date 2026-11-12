use serde_json::json;

use crate::layout::{ImageProcessorConfig, ImageTags, VisionConfig, VisionPipeline};

/// The fields a checkpoint of this contract declares; the `model_type`
/// says nothing to discovery, the shape of the configuration does.
fn config() -> serde_json::Value {
    json!({
        "downsample_ratio": 0.5,
        "ps_version": "v2",
        "vit_hidden_size": 1280,
        "projector_hidden_size": 20480,
        "img_context_token_id": 18,
        "image_tag_type": "internvl",
        "llm_config": {"hidden_size": 4096, "num_nextn_predict_layers": 1},
        "vision_config": {
            "model_type": "any encoder",
            "hidden_size": 1280,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "patch_size": 16,
            "mlp_ratio": 4.0,
            "num_cls_tokens": 3,
            "num_registers": 7,
            "max_img_size": 2048,
            "layer_norm_eps": 1e-06,
            "hidden_act": "gelu",
            "num_channels": 3
        }
    })
}

#[test]
fn a_prefix_token_encoder_with_a_pixel_shuffle_is_discovered() -> crate::error::Result<()> {
    let Some(VisionConfig::PixelShuffleEncoder(vision)) = VisionConfig::from_value(&config())?
    else {
        return Err(crate::error::ModelsError::InvalidConfig("contract not discovered".into()));
    };
    assert_eq!((vision.hidden_size, vision.intermediate_size), (1280, 5120));
    assert_eq!(
        (vision.prefix_tokens, vision.position_grid, vision.shuffle_factor),
        (10, 128, 2)
    );
    assert_eq!((vision.projector_hidden_size, vision.output_hidden_size), (20480, 4096));
    assert!(vision.final_layer_norm);
    assert_eq!((vision.image_token_id, vision.image_tags), (18, ImageTags::Img));
    assert_eq!((ImageTags::Img.start(), ImageTags::Img.end()), ("<img>", "</img>"));
    Ok(())
}

#[test]
fn an_unknown_shuffle_or_tag_scheme_is_not_guessed() {
    let mut transposed = config();
    transposed["ps_version"] = json!("v1");
    assert!(VisionConfig::from_value(&transposed).is_err());
    let mut tags = config();
    tags["image_tag_type"] = json!("other");
    assert!(VisionConfig::from_value(&tags).is_err());
}

#[test]
fn a_decoder_without_prediction_layers_has_no_final_norm() -> crate::error::Result<()> {
    let mut plain = config();
    plain["llm_config"] = json!({"hidden_size": 4096});
    let Some(VisionConfig::PixelShuffleEncoder(vision)) = VisionConfig::from_value(&plain)? else {
        return Err(crate::error::ModelsError::InvalidConfig("contract not discovered".into()));
    };
    assert!(!vision.final_layer_norm);
    Ok(())
}

#[test]
fn the_processor_budget_is_the_context_share_within_bounds() -> crate::error::Result<()> {
    let processor = json!({
        "downsample_ratio": 0.5,
        "max_model_len": 16384,
        "max_num_patches": 13312,
        "min_num_patches": 1024,
        "norm_mean": [0.481_454_66, 0.457_827_5, 0.408_210_73],
        "norm_std": [0.268_629_54, 0.261_302_58, 0.275_777_11],
        "patch_size": 16,
        "rescale_factor": 0.003_921_568_627_450_98
    });
    let ImageProcessorConfig::PixelShuffle(config) =
        ImageProcessorConfig::from_value(&processor, VisionPipeline::PixelShuffleEncoder)?
    else {
        return Err(crate::error::ModelsError::InvalidConfig("processor not parsed".into()));
    };
    assert_eq!((config.shuffle_factor, config.patch_budget()), (2, 13312));
    Ok(())
}
