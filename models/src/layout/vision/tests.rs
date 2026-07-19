use std::fs;

use serde_json::json;

use super::*;
use crate::{
    error::{ModelsError, Result},
    layout::ModelLayout,
};

#[test]
fn discovers_pooled_contract_despite_misleading_model_identity() -> Result<()> {
    let value = json!({
        "architectures": ["SpatialMergeForConditionalGeneration"],
        "model_type": "unrelated_model_family",
        "image_token_id": 258_880,
        "boi_token_id": 255_999,
        "eoi_token_id": 258_882,
        "vision_soft_tokens_per_image": 280,
        "text_config": {
            "hidden_size": 2816,
            "use_bidirectional_attention": "vision"
        },
        "vision_config": {
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "head_dim": 72,
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "position_embedding_size": 10240,
            "rms_norm_eps": 0.000_001,
            "rope_parameters": {"rope_theta": 100.0},
            "hidden_activation": "gelu_pytorch_tanh",
            "use_clipped_linears": true,
            "standardize": true
        }
    });

    let Some(config) = VisionConfig::from_value(&value)? else {
        return Err(missing("pooled vision config"));
    };
    let VisionConfig::PooledEncoder(config) = config else {
        return Err(missing("pooled vision pipeline"));
    };
    assert_eq!(config.hidden_size, 1152);
    assert_eq!(config.output_hidden_size, 2816);
    assert_eq!(config.head_dim, 72);
    assert_eq!(config.num_key_value_heads, 16);
    assert!((config.rope_theta - 100.0).abs() < f64::EPSILON);
    assert!(config.use_clipped_linears);
    assert_eq!(config.soft_tokens_per_image, 280);
    assert!(config.bidirectional_image_attention);
    Ok(())
}

#[test]
fn discovers_spatial_merge_contract_despite_misleading_model_identity() -> Result<()> {
    let value = json!({
        "architectures": ["PooledEncoderForConditionalGeneration"],
        "model_type": "another_unrelated_family",
        "image_token_id": 248_056,
        "vision_start_token_id": 248_053,
        "vision_end_token_id": 248_054,
        "text_config": {
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10]
            }
        },
        "vision_config": {
            "model_type": "also_ignored",
            "depth": 27,
            "hidden_size": 1152,
            "out_hidden_size": 2048,
            "intermediate_size": 4304,
            "num_heads": 16,
            "in_channels": 3,
            "patch_size": [16, 16],
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "num_position_embeddings": 2304,
            "hidden_act": "gelu_pytorch_tanh"
        }
    });

    let Some(config) = VisionConfig::from_value(&value)? else {
        return Err(missing("spatial merge vision config"));
    };
    let VisionConfig::SpatialMergeEncoder(config) = config else {
        return Err(missing("spatial merge vision pipeline"));
    };
    assert_eq!(config.patch_size, 16);
    assert_eq!(config.output_hidden_size, 2048);
    assert_eq!(config.mrope_sections, vec![11, 11, 10]);
    assert!(config.mrope_interleaved);
    Ok(())
}

#[test]
fn reads_nested_pooled_processor_and_flat_spatial_merge_processor() -> Result<()> {
    let pooled = layout_with_processor(
        "pooled",
        "processor_config.json",
        &json!({
            "image_processor": {
                "patch_size": 16,
                "pooling_kernel_size": 3,
                "max_soft_tokens": 280,
                "rescale_factor": 0.003_921_568_627_450_98,
                "do_resize": true,
                "do_rescale": true,
                "do_normalize": false
            }
        }),
    )?;
    let Some(processor) =
        ImageProcessorConfig::from_layout(&pooled, VisionPipeline::PooledEncoder)?
    else {
        return Err(missing("pooled image processor"));
    };
    assert!(matches!(
        processor,
        ImageProcessorConfig::Pooled(PooledImageProcessorConfig {
            max_soft_tokens: 280,
            do_normalize: false,
            ..
        })
    ));
    remove_layout(pooled)?;

    let spatial_merge = layout_with_processor(
        "spatial-merge",
        "preprocessor_config.json",
        &json!({
            "size": {"shortest_edge": 65_536, "longest_edge": 16_777_216},
            "patch_size": 16,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5]
        }),
    )?;
    let Some(processor) =
        ImageProcessorConfig::from_layout(&spatial_merge, VisionPipeline::SpatialMergeEncoder)?
    else {
        return Err(missing("spatial merge image processor"));
    };
    assert!(matches!(
        processor,
        ImageProcessorConfig::SpatialMerge(SpatialMergeImageProcessorConfig {
            min_pixels: 65536,
            max_pixels: 16_777_216,
            ..
        })
    ));
    remove_layout(spatial_merge)?;
    Ok(())
}

#[test]
fn discovers_official_qwen_video_preprocessor_filename() -> Result<()> {
    let root = temp_root("qwen-video");
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    fs::write(root.join("video_preprocessor_config.json"), "{}")?;
    let layout = ModelLayout::inspect(&root)?;

    assert_eq!(
        layout.video_processor_config_path.as_deref(),
        Some(root.join("video_preprocessor_config.json").as_path())
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

fn layout_with_processor(
    name: &str,
    filename: &str,
    processor: &serde_json::Value,
) -> Result<ModelLayout> {
    let root = temp_root(name);
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    fs::write(root.join(filename), serde_json::to_vec(processor)?)?;
    ModelLayout::inspect(root)
}

fn remove_layout(layout: ModelLayout) -> Result<()> {
    fs::remove_dir_all(layout.root)?;
    Ok(())
}

fn temp_root(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "libmir-vision-{name}-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ))
}

fn missing(item: &str) -> ModelsError {
    ModelsError::InvalidConfig(format!("missing test {item}"))
}
