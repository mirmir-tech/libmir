use super::{PixelShufflePromptTokens, target};
use crate::layout::PixelShuffleImageProcessorConfig;

fn processor() -> PixelShuffleImageProcessorConfig {
    PixelShuffleImageProcessorConfig {
        patch_size: 16,
        shuffle_factor: 2,
        min_patches: 1024,
        max_patches: 13312,
        context_patches: Some((16384 - 4) * 4),
        rescale_factor: 1.0 / 255.0,
        image_mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
        image_std: [0.268_629_54, 0.261_302_58, 0.275_777_11],
    }
}

/// Grids the processor's own rule gives (its Python function run on these
/// sizes): `(height, width, budget, (rows, cols))`.
const GRIDS: [(usize, usize, usize, (usize, usize)); 10] = [
    (400, 640, 13312, (26, 40)),
    (700, 300, 13312, (50, 22)),
    (1080, 1920, 13312, (68, 120)),
    (64, 64, 13312, (32, 32)),
    (48, 48, 13312, (32, 32)),
    (3000, 4000, 13312, (100, 132)),
    (1000, 17, 13312, (180, 6)),
    (400, 640, 2000, (26, 40)),
    (400, 640, 512, (18, 28)),
    (64, 8192, 13312, (4, 512)),
];

#[test]
fn the_patch_grid_follows_the_processors_rule() -> crate::error::Result<()> {
    let config = processor();
    for (height, width, budget, expected) in GRIDS {
        assert_eq!(target::patches(height, width, &config, budget)?, expected, "{height}x{width}");
    }
    Ok(())
}

#[test]
fn halves_round_to_the_even_neighbour() {
    assert!((target::round(25.5) - 26.0).abs() < f64::EPSILON);
    assert!((target::round(40.5) - 40.0).abs() < f64::EPSILON);
    assert!((target::round(3.4) - 3.0).abs() < f64::EPSILON);
}

#[test]
fn patches_are_packed_row_major_channel_y_x() -> crate::error::Result<()> {
    let mut config = processor();
    config.min_patches = 4;
    config.max_patches = 4;
    config.context_patches = None;
    config.image_mean = [0.0; 3];
    config.image_std = [1.0; 3];
    config.rescale_factor = 1.0;
    // 32×32 pixels are 2×2 patches of 16, no resize.
    let rgb: Vec<u8> = (0..32_u8).flat_map(|y| (0..32_u8).flat_map(move |x| [x, y, 7])).collect();
    let image = config.preprocess_rgb(&rgb, 32, 32)?;
    assert_eq!((image.grid_height, image.grid_width, image.soft_tokens), (2, 2, 1));
    let per_patch = 3 * 16 * 16;
    // Patch (row 0, col 1), red channel, y 0: x runs 16..32.
    assert!((image.patches[per_patch] - 16.0).abs() < 1e-6);
    // Same patch, green channel, y 3: the pixel row is 3.
    assert!((image.patches[per_patch + 256 + 3 * 16] - 3.0).abs() < 1e-6);
    // Patch (row 1, col 0), blue channel: constant 7.
    assert!((image.patches[2 * per_patch + 2 * 256] - 7.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn the_image_token_becomes_tags_around_one_token_per_soft_token() -> crate::error::Result<()> {
    let image = super::PixelShufflePreprocessedImage {
        patches: Vec::new(),
        grid_height: 4,
        grid_width: 2,
        soft_tokens: 2,
    };
    let tokens = PixelShufflePromptTokens::prepare(&[5, 18, 6], &image, 18, (19, 20))?;
    assert_eq!(tokens.token_ids, [5, 19, 18, 18, 20, 6]);
    assert_eq!((tokens.image_start, tokens.image_end), (2, 4));
    assert!(PixelShufflePromptTokens::prepare(&[18, 18], &image, 18, (19, 20)).is_err());
    Ok(())
}

/// Writes the patches of `PIXEL_SHUFFLE_IMAGE` as little-endian f32 to
/// `PIXEL_SHUFFLE_OUTPUT`, for comparison with a reference preprocessor.
#[test]
#[ignore = "needs PIXEL_SHUFFLE_IMAGE and PIXEL_SHUFFLE_OUTPUT"]
fn writes_patches_for_reference_comparison() -> crate::error::Result<()> {
    let input = std::env::var("PIXEL_SHUFFLE_IMAGE").unwrap_or_default();
    let output = std::env::var("PIXEL_SHUFFLE_OUTPUT").unwrap_or_default();
    let image = processor().preprocess_encoded(&std::fs::read(input)?)?;
    let bytes: Vec<u8> = image.patches.iter().flat_map(|value| value.to_le_bytes()).collect();
    std::fs::write(output, bytes)?;
    Ok(())
}
