use image::{RgbImage, imageops::FilterType};

use crate::{
    error::{ModelsError, Result},
    layout::PooledImageProcessorConfig,
};

mod prompt;

pub use prompt::PooledPromptTokens;

const SUPPORTED_SOFT_TOKENS: [usize; 5] = [70, 140, 280, 560, 1_120];

#[derive(Debug, Clone, PartialEq)]
pub struct PooledPreprocessedImage {
    pub patches: Vec<f32>,
    pub position_ids: Vec<u32>,
    pub grid_height: usize,
    pub grid_width: usize,
    pub soft_tokens: usize,
}

impl PooledImageProcessorConfig {
    pub fn preprocess_encoded(&self, encoded: &[u8]) -> Result<PooledPreprocessedImage> {
        let image = image::load_from_memory(encoded)?.to_rgb8();
        self.preprocess_image(&image, None)
    }

    pub fn preprocess_encoded_with_patch_limit(
        &self,
        encoded: &[u8],
        max_patches: usize,
    ) -> Result<PooledPreprocessedImage> {
        let image = image::load_from_memory(encoded)?.to_rgb8();
        self.preprocess_image(&image, Some(max_patches))
    }

    pub fn preprocess_rgb(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
    ) -> Result<PooledPreprocessedImage> {
        let image = RgbImage::from_raw(u32::try_from(width)?, u32::try_from(height)?, rgb.to_vec())
            .ok_or_else(|| invalid("RGB byte length does not match image dimensions"))?;
        self.preprocess_image(&image, None)
    }

    fn preprocess_image(
        &self,
        image: &RgbImage,
        max_patches: Option<usize>,
    ) -> Result<PooledPreprocessedImage> {
        self.validate()?;
        let width = usize::try_from(image.width())?;
        let height = usize::try_from(image.height())?;
        let (target_height, target_width) = if self.do_resize {
            target_size(height, width, self, max_patches)?
        } else {
            (height, width)
        };
        let side_multiple = self
            .patch_size
            .checked_mul(self.pooling_kernel_size)
            .ok_or_else(|| invalid("pooled vision image side multiple overflowed"))?;
        if target_height == 0
            || target_width == 0
            || !target_height.is_multiple_of(side_multiple)
            || !target_width.is_multiple_of(side_multiple)
        {
            return Err(invalid(format!(
                "pooled vision image size {target_height}x{target_width} must use nonzero sides divisible by {side_multiple}"
            )));
        }
        let resized = if target_height == height && target_width == width {
            image.clone()
        } else {
            image::imageops::resize(
                image,
                u32::try_from(target_width)?,
                u32::try_from(target_height)?,
                FilterType::CatmullRom,
            )
        };
        let output = patchify(&resized, self)?;
        if max_patches.is_some_and(|limit| output.grid_height * output.grid_width > limit) {
            return Err(invalid(
                "pooled vision image exceeds the runtime patch budget with resizing disabled",
            ));
        }
        Ok(output)
    }

    fn validate(&self) -> Result<()> {
        if self.patch_size == 0
            || self.pooling_kernel_size == 0
            || !SUPPORTED_SOFT_TOKENS.contains(&self.max_soft_tokens)
            || self.do_normalize
            || !self.rescale_factor.is_finite()
        {
            return Err(invalid(format!(
                "unsupported pooled vision image processor: patch={}, pool={}, max_soft_tokens={}, normalize={}, rescale_factor={}",
                self.patch_size,
                self.pooling_kernel_size,
                self.max_soft_tokens,
                self.do_normalize,
                self.rescale_factor
            )));
        }
        Ok(())
    }
}

fn target_size(
    height: usize,
    width: usize,
    config: &PooledImageProcessorConfig,
    runtime_max_patches: Option<usize>,
) -> Result<(usize, usize)> {
    if height == 0 || width == 0 {
        return Err(invalid("cannot resize an empty pooled vision image"));
    }
    let kernel_squared = config
        .pooling_kernel_size
        .checked_mul(config.pooling_kernel_size)
        .ok_or_else(|| invalid("pooled vision pooling kernel overflowed"))?;
    let checkpoint_max_patches = config
        .max_soft_tokens
        .checked_mul(kernel_squared)
        .ok_or_else(|| invalid("pooled vision patch budget overflowed"))?;
    let max_patches = runtime_max_patches
        .map_or(checkpoint_max_patches, |limit| limit.min(checkpoint_max_patches));
    if max_patches < kernel_squared {
        return Err(invalid("pooled vision runtime patch budget is too small"));
    }
    let patch_squared = config
        .patch_size
        .checked_mul(config.patch_size)
        .ok_or_else(|| invalid("pooled vision patch area overflowed"))?;
    let total_pixels = height.checked_mul(width).ok_or_else(|| invalid("image area overflowed"))?;
    let target_pixels = max_patches
        .checked_mul(patch_squared)
        .ok_or_else(|| invalid("target image area overflowed"))?;
    let factor = (target_pixels.to_string().parse::<f64>()?
        / total_pixels.to_string().parse::<f64>()?)
    .sqrt();
    let side_multiple = config.patch_size * config.pooling_kernel_size;
    let mut target_height = ((factor * height.to_string().parse::<f64>()?)
        / side_multiple.to_string().parse::<f64>()?)
    .floor()
    .to_string()
    .parse::<usize>()?
        * side_multiple;
    let mut target_width = ((factor * width.to_string().parse::<f64>()?)
        / side_multiple.to_string().parse::<f64>()?)
    .floor()
    .to_string()
    .parse::<usize>()?
        * side_multiple;
    let max_side = (max_patches / kernel_squared) * side_multiple;
    if target_height == 0 && target_width == 0 {
        return Err(invalid("pooled vision image would resize to zero in both dimensions"));
    }
    if target_height == 0 {
        target_height = side_multiple;
        target_width = (width / height).saturating_mul(side_multiple).min(max_side);
    } else if target_width == 0 {
        target_width = side_multiple;
        target_height = (height / width).saturating_mul(side_multiple).min(max_side);
    }
    if target_height * target_width > target_pixels {
        return Err(invalid("pooled vision resized image exceeds its patch budget"));
    }
    Ok((target_height, target_width))
}

fn patchify(
    image: &RgbImage,
    config: &PooledImageProcessorConfig,
) -> Result<PooledPreprocessedImage> {
    let width = usize::try_from(image.width())?;
    let height = usize::try_from(image.height())?;
    let grid_width = width / config.patch_size;
    let grid_height = height / config.patch_size;
    let patch_values = 3 * config.patch_size * config.patch_size;
    let mut patches = Vec::with_capacity(grid_width * grid_height * patch_values);
    let mut position_ids = Vec::with_capacity(grid_width * grid_height * 2);
    let scale = if config.do_rescale {
        config.rescale_factor.to_string().parse::<f32>()?
    } else {
        1.0
    };
    for patch_y in 0..grid_height {
        for patch_x in 0..grid_width {
            position_ids.extend([u32::try_from(patch_x)?, u32::try_from(patch_y)?]);
            for inner_y in 0..config.patch_size {
                for inner_x in 0..config.patch_size {
                    let pixel = image.get_pixel(
                        u32::try_from(patch_x * config.patch_size + inner_x)?,
                        u32::try_from(patch_y * config.patch_size + inner_y)?,
                    );
                    patches.extend(pixel.0.map(|channel| f32::from(channel) * scale));
                }
            }
        }
    }
    let pooled_height = grid_height / config.pooling_kernel_size;
    let pooled_width = grid_width / config.pooling_kernel_size;
    let soft_tokens = pooled_height * pooled_width;
    Ok(PooledPreprocessedImage {
        patches,
        position_ids,
        grid_height,
        grid_width,
        soft_tokens,
    })
}

fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}

#[cfg(test)]
mod tests;
