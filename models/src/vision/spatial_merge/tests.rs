use super::*;
use crate::layout::SpatialMergeVisionConfig;

#[test]
fn duplicates_the_image_along_the_temporal_patch_axis() -> Result<()> {
    let image = processor().preprocess_rgb(&[10, 20, 30, 40, 50, 60], 2, 1)?;
    assert_eq!((image.grid_t, image.grid_height, image.grid_width), (1, 1, 2));
    assert_eq!(image.soft_tokens, 2);
    assert_eq!(
        image.patches,
        [10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0, 50.0, 50.0, 60.0, 60.0]
    );
    Ok(())
}

#[test]
fn builds_spatial_mrope_positions_and_decode_delta() -> Result<()> {
    let image = SpatialMergePreprocessedImage {
        patches: Vec::new(),
        grid_t: 1,
        grid_height: 2,
        grid_width: 2,
        soft_tokens: 4,
    };
    let config = SpatialMergeVisionConfig {
        hidden_size: 8,
        output_hidden_size: 8,
        intermediate_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        in_channels: 3,
        patch_size: 1,
        temporal_patch_size: 2,
        spatial_merge_size: 1,
        num_position_embeddings: 4,
        hidden_activation: "gelu_pytorch_tanh".into(),
        image_token_id: 10,
        vision_start_token_id: 11,
        vision_end_token_id: 12,
        mrope_interleaved: true,
        mrope_sections: vec![1, 1, 2],
    };
    let prompt = SpatialMergePromptTokens::prepare(&[7, 10, 8], &image, &config)?;
    assert_eq!(prompt.token_ids, [7, 11, 10, 10, 10, 10, 12, 8]);
    assert_eq!(prompt.position_delta, -2);
    assert_eq!(
        prompt.position_ids,
        [0, 1, 2, 2, 2, 2, 4, 5, 0, 1, 2, 2, 3, 3, 4, 5, 0, 1, 2, 3, 2, 3, 4, 5]
    );
    Ok(())
}

#[test]
fn runtime_pixel_limit_reduces_checkpoint_budget() -> Result<()> {
    let (height, width) = smart_resize(600, 1_200, 2, 4, 400)?;

    assert!(height * width <= 400);
    assert_eq!(width / height, 2);
    Ok(())
}

fn processor() -> SpatialMergeImageProcessorConfig {
    SpatialMergeImageProcessorConfig {
        patch_size: 1,
        temporal_patch_size: 2,
        spatial_merge_size: 1,
        min_pixels: 1,
        max_pixels: 16,
        rescale_factor: 1.0,
        image_mean: [0.0; 3],
        image_std: [1.0; 3],
        do_resize: true,
        do_rescale: false,
        do_normalize: false,
    }
}
