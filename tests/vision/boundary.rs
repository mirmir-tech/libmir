use std::{io::Cursor, path::PathBuf};

use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use libmir::{
    ChatCompletionRequest, ChatMessage, GenerationOverrides, IMAGE_PLACEHOLDER, Library,
    PreparedVisionPrompt, Result, RuntimeConfig, SamplingLogits,
};
use models::layout::{ImageProcessorConfig, VisionConfig};

#[test]
#[ignore = "loads a real vision checkpoint; set MODEL"]
fn executes_the_maximum_configured_geometry_and_downscales_the_next() -> Result<()> {
    let path = required_path("MODEL")?;
    let inspected = libmir::ModelDescriptor::inspect(&path, GenerationOverrides::default())?;
    let (side, patch_tokens, heads) = boundary_geometry(&inspected)?;
    let mut config = RuntimeConfig::default();
    config.kv_cache.block_count = 128;
    config.vision.max_pixels = Some(side.checked_mul(side).ok_or_else(overflow)?);
    config.vision.attention_budget_bytes = Some(attention_bytes(patch_tokens, heads)?);
    let model =
        Library::new(config).load(path, GenerationOverrides::default(), &mut |_event| {})?;
    let request = request(&model);
    let boundary = model.prepare_image(&request, &solid_png(side)?)?;
    let oversized = model.prepare_image(&request, &solid_png(side + merge_factor(&model)?)?)?;
    let boundary_image = spatial_image(&boundary)?;
    let oversized_image = spatial_image(&oversized)?;
    assert_eq!(boundary_image.grid_height * boundary_image.grid_width, patch_tokens);
    assert_eq!(boundary_image, oversized_image);
    let boundary_output =
        model
            .session()
            .prefill_vision(&boundary, SamplingLogits::None, &mut |_event| {})?;
    let oversized_output =
        model
            .session()
            .prefill_vision(&oversized, SamplingLogits::None, &mut |_event| {})?;
    assert_eq!(boundary_output.accepted_tokens, oversized_output.accepted_tokens);
    assert_eq!(boundary_output.next_token, oversized_output.next_token);
    assert!(boundary_output.next_token.is_some());
    Ok(())
}

fn required_path(name: &'static str) -> Result<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .ok_or(libmir::Error::MissingEnvironment(name))
}

fn boundary_geometry(descriptor: &libmir::ModelDescriptor) -> Result<(usize, usize, usize)> {
    let VisionConfig::SpatialMergeEncoder(vision) = descriptor
        .vision()
        .ok_or_else(|| backend_error("checkpoint has no vision contract"))?
    else {
        return Err(backend_error("checkpoint does not use spatial-merge vision").into());
    };
    let ImageProcessorConfig::SpatialMerge(processor) = descriptor
        .image_processor()
        .ok_or_else(|| backend_error("checkpoint has no image processor"))?
    else {
        return Err(backend_error("checkpoint does not use a spatial-merge processor").into());
    };
    let factor = vision.patch_size.checked_mul(vision.spatial_merge_size).ok_or_else(overflow)?;
    let side = aligned_square_side(processor.min_pixels, factor)?;
    let patches_per_side = side / vision.patch_size;
    let patch_tokens = patches_per_side.checked_mul(patches_per_side).ok_or_else(overflow)?;
    Ok((side, patch_tokens, vision.num_attention_heads))
}

fn aligned_square_side(minimum_pixels: usize, factor: usize) -> Result<usize> {
    let mut side = factor;
    while side.checked_mul(side).ok_or_else(overflow)? < minimum_pixels {
        side = side.checked_add(factor).ok_or_else(overflow)?;
    }
    Ok(side)
}

fn attention_bytes(patches: usize, heads: usize) -> Result<u64> {
    let bytes = 4_u128
        .checked_mul(patches as u128)
        .and_then(|value| value.checked_mul(patches as u128))
        .and_then(|value| value.checked_mul(heads as u128))
        .ok_or_else(overflow)?;
    match u64::try_from(bytes) {
        Ok(bytes) => Ok(bytes),
        Err(error) => Err(backend_error(&error.to_string()).into()),
    }
}

fn merge_factor(model: &libmir::Model) -> Result<usize> {
    let VisionConfig::SpatialMergeEncoder(vision) = model
        .descriptor()
        .vision()
        .ok_or_else(|| backend_error("loaded model has no vision contract"))?
    else {
        return Err(backend_error("loaded model does not use spatial-merge vision").into());
    };
    vision
        .patch_size
        .checked_mul(vision.spatial_merge_size)
        .ok_or_else(|| overflow().into())
}

fn spatial_image(
    prepared: &PreparedVisionPrompt,
) -> Result<&models::vision::SpatialMergePreprocessedImage> {
    match prepared {
        PreparedVisionPrompt::SpatialMerge { image, .. } => Ok(image),
        PreparedVisionPrompt::Pooled { .. } => {
            Err(backend_error("prepared prompt does not use spatial-merge vision").into())
        },
    }
}

fn solid_png(side: usize) -> Result<Vec<u8>> {
    let side = match u32::try_from(side) {
        Ok(side) => side,
        Err(error) => return Err(backend_error(&error.to_string()).into()),
    };
    let image = RgbImage::from_pixel(side, side, Rgb([0, 0, 0]));
    let mut encoded = Cursor::new(Vec::new());
    if let Err(error) = DynamicImage::ImageRgb8(image).write_to(&mut encoded, ImageFormat::Png) {
        return Err(backend_error(&error.to_string()).into());
    }
    Ok(encoded.into_inner())
}

fn request(model: &libmir::Model) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: model.handle().id.clone(),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: format!("{IMAGE_PLACEHOLDER}Describe the image."),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        tools: Vec::new(),
        tool_choice: None,
        stream: false,
        max_tokens: Some(1),
        min_tokens: None,
        ignore_eos: None,
        temperature: Some(0.0),
        top_p: Some(1.0),
        top_k: Some(0),
        repetition_penalty: Some(1.0),
        seed: Some(7),
    }
}

fn overflow() -> runtime::RuntimeError {
    backend_error("vision boundary geometry overflowed")
}

fn backend_error(message: &str) -> runtime::RuntimeError {
    runtime::RuntimeError::Backend(message.into())
}
