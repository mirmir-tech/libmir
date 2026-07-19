use image::{RgbImage, imageops::FilterType};

use crate::{
    error::{ModelsError, Result},
    layout::SpatialMergeImageProcessorConfig,
};

mod prompt;

pub use prompt::SpatialMergePromptTokens;

#[derive(Debug, Clone, PartialEq)]
pub struct SpatialMergePreprocessedImage {
    pub patches: Vec<f32>,
    pub grid_t: usize,
    pub grid_height: usize,
    pub grid_width: usize,
    pub soft_tokens: usize,
}

impl SpatialMergeImageProcessorConfig {
    pub fn preprocess_encoded(&self, encoded: &[u8]) -> Result<SpatialMergePreprocessedImage> {
        self.preprocess_image(&image::load_from_memory(encoded)?.to_rgb8(), None)
    }

    pub fn preprocess_encoded_with_max_pixels(
        &self,
        encoded: &[u8],
        max_pixels: usize,
    ) -> Result<SpatialMergePreprocessedImage> {
        self.preprocess_image(&image::load_from_memory(encoded)?.to_rgb8(), Some(max_pixels))
    }

    pub fn preprocess_rgb(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
    ) -> Result<SpatialMergePreprocessedImage> {
        let image = RgbImage::from_raw(u32::try_from(width)?, u32::try_from(height)?, rgb.to_vec())
            .ok_or_else(|| invalid("RGB byte length does not match image dimensions"))?;
        self.preprocess_image(&image, None)
    }

    fn preprocess_image(
        &self,
        image: &RgbImage,
        runtime_max_pixels: Option<usize>,
    ) -> Result<SpatialMergePreprocessedImage> {
        self.validate()?;
        let original = (usize::try_from(image.height())?, usize::try_from(image.width())?);
        let factor = self.patch_size * self.spatial_merge_size;
        let maximum =
            runtime_max_pixels.map_or(self.max_pixels, |limit| limit.min(self.max_pixels));
        if maximum < factor * factor {
            return Err(invalid("spatial-merge vision runtime pixel budget is too small"));
        }
        let minimum = self.min_pixels.min(maximum);
        let (height, width) = if self.do_resize {
            smart_resize(original.0, original.1, factor, minimum, maximum)?
        } else if original.0.saturating_mul(original.1) > maximum {
            return Err(invalid(
                "spatial-merge vision image exceeds the runtime pixel budget with resizing disabled",
            ));
        } else if original.0.is_multiple_of(factor) && original.1.is_multiple_of(factor) {
            original
        } else {
            return Err(invalid(format!(
                "spatial-merge vision image sides must be divisible by {factor} when resizing is disabled"
            )));
        };
        let resized =
            if image.height() == u32::try_from(height)? && image.width() == u32::try_from(width)? {
                image.clone()
            } else {
                image::imageops::resize(
                    image,
                    u32::try_from(width)?,
                    u32::try_from(height)?,
                    FilterType::CatmullRom,
                )
            };
        patchify(&resized, self)
    }

    fn validate(&self) -> Result<()> {
        if self.patch_size == 0
            || self.temporal_patch_size == 0
            || self.spatial_merge_size == 0
            || self.min_pixels == 0
            || self.max_pixels < self.min_pixels
            || self.image_std.iter().any(|value| !value.is_finite() || *value == 0.0)
        {
            return Err(invalid("unsupported spatial-merge vision image processor configuration"));
        }
        Ok(())
    }
}

fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    minimum: usize,
    maximum: usize,
) -> Result<(usize, usize)> {
    if height == 0 || width == 0 || height.max(width) > 200 * height.min(width) {
        return Err(invalid(
            "spatial-merge vision image is empty or exceeds the 200:1 aspect ratio",
        ));
    }
    let rounded = |value: usize| ((value + factor / 2) / factor).max(1) * factor;
    let mut target_height = rounded(height);
    let mut target_width = rounded(width);
    let area = height.checked_mul(width).ok_or_else(|| invalid("image area overflowed"))?;
    if target_height * target_width > maximum {
        let scale = (area.to_string().parse::<f64>()? / maximum.to_string().parse::<f64>()?).sqrt();
        target_height = float_usize(
            ((height.to_string().parse::<f64>()? / scale) / factor.to_string().parse::<f64>()?)
                .floor()
                .max(1.0),
        )? * factor;
        target_width = float_usize(
            ((width.to_string().parse::<f64>()? / scale) / factor.to_string().parse::<f64>()?)
                .floor()
                .max(1.0),
        )? * factor;
    } else if target_height * target_width < minimum {
        let scale = (minimum.to_string().parse::<f64>()? / area.to_string().parse::<f64>()?).sqrt();
        target_height = float_usize(
            ((height.to_string().parse::<f64>()? * scale) / factor.to_string().parse::<f64>()?)
                .ceil(),
        )? * factor;
        target_width = float_usize(
            ((width.to_string().parse::<f64>()? * scale) / factor.to_string().parse::<f64>()?)
                .ceil(),
        )? * factor;
    }
    Ok((target_height, target_width))
}

fn patchify(
    image: &RgbImage,
    config: &SpatialMergeImageProcessorConfig,
) -> Result<SpatialMergePreprocessedImage> {
    let grid_height = usize::try_from(image.height())? / config.patch_size;
    let grid_width = usize::try_from(image.width())? / config.patch_size;
    let merge = config.spatial_merge_size;
    let patch_width = 3 * config.temporal_patch_size * config.patch_size * config.patch_size;
    let mut patches = Vec::with_capacity(grid_height * grid_width * patch_width);
    for block_y in 0..grid_height / merge {
        for block_x in 0..grid_width / merge {
            for merge_y in 0..merge {
                for merge_x in 0..merge {
                    append_patch(
                        &mut patches,
                        image,
                        block_y * merge + merge_y,
                        block_x * merge + merge_x,
                        config,
                    )?;
                }
            }
        }
    }
    let soft_tokens = (grid_height / merge) * (grid_width / merge);
    Ok(SpatialMergePreprocessedImage {
        patches,
        grid_t: 1,
        grid_height,
        grid_width,
        soft_tokens,
    })
}

fn append_patch(
    output: &mut Vec<f32>,
    image: &RgbImage,
    patch_y: usize,
    patch_x: usize,
    config: &SpatialMergeImageProcessorConfig,
) -> Result<()> {
    for channel in 0..3 {
        for _frame in 0..config.temporal_patch_size {
            for inner_y in 0..config.patch_size {
                for inner_x in 0..config.patch_size {
                    let pixel = image.get_pixel(
                        u32::try_from(patch_x * config.patch_size + inner_x)?,
                        u32::try_from(patch_y * config.patch_size + inner_y)?,
                    );
                    let mut value = f64::from(pixel[channel]);
                    if config.do_rescale {
                        value *= config.rescale_factor;
                    }
                    if config.do_normalize {
                        value = (value - config.image_mean[channel]) / config.image_std[channel];
                    }
                    output.push(value.to_string().parse()?);
                }
            }
        }
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}

fn float_usize(value: f64) -> Result<usize> {
    Ok(value.to_string().parse()?)
}

#[cfg(test)]
mod tests;
