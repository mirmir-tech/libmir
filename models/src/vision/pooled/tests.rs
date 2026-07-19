use super::*;

#[test]
fn patchifies_rgb_in_reference_order_with_xy_positions() -> Result<()> {
    let config = config(false);
    let image = config.preprocess_rgb(&[255, 0, 127, 0, 255, 64], 2, 1)?;

    assert_eq!(image.grid_height, 1);
    assert_eq!(image.grid_width, 2);
    assert_eq!(image.soft_tokens, 2);
    assert_eq!(image.position_ids, vec![0, 0, 1, 0]);
    assert_close(&image.patches, &[1.0, 0.0, 127.0 / 255.0, 0.0, 1.0, 64.0 / 255.0]);
    Ok(())
}

#[test]
fn preserves_aspect_ratio_and_pooling_divisibility() -> Result<()> {
    let config = config(true);
    let (height, width) = target_size(600, 1_200, &config, None)?;
    let side = config.patch_size * config.pooling_kernel_size;
    let patches = (height / config.patch_size) * (width / config.patch_size);

    assert_eq!(height % side, 0);
    assert_eq!(width % side, 0);
    assert!(patches <= config.max_soft_tokens * config.pooling_kernel_size.pow(2));
    assert_eq!(width / height, 2);
    Ok(())
}

#[test]
fn runtime_patch_limit_reduces_checkpoint_budget() -> Result<()> {
    let config = config(true);
    let (height, width) = target_size(600, 1_200, &config, Some(20))?;
    let patches = (height / config.patch_size) * (width / config.patch_size);

    assert!(patches <= 20);
    assert_eq!(width / height, 2);
    Ok(())
}

fn config(do_resize: bool) -> PooledImageProcessorConfig {
    PooledImageProcessorConfig {
        patch_size: 1,
        pooling_kernel_size: 1,
        max_soft_tokens: 70,
        rescale_factor: 1.0 / 255.0,
        do_resize,
        do_rescale: true,
        do_normalize: false,
    }
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().zip(expected).all(|(left, right)| (left - right).abs() < 1.0e-6));
}
