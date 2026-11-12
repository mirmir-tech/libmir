//! Preprocessing for the pixel-shuffle encoder: one tile at the image's
//! aspect ratio, sized to the patch budget, resized in floating point and
//! normalized, packed patch by patch (row-major grid, each patch channel,
//! y, x).

use image::RgbImage;

use crate::{
    error::{ModelsError, Result},
    layout::PixelShuffleImageProcessorConfig,
};

mod numeric;
mod prompt;
mod resize;
mod target;
#[cfg(test)]
mod tests;

pub use prompt::PixelShufflePromptTokens;
use resize::Planar;

#[derive(Debug, Clone, PartialEq)]
pub struct PixelShufflePreprocessedImage {
    pub patches: Vec<f32>,
    /// Patch rows and columns before the shuffle.
    pub grid_height: usize,
    pub grid_width: usize,
    /// Tokens after the shuffle folds `factor²` patches into one.
    pub soft_tokens: usize,
}

impl PixelShuffleImageProcessorConfig {
    pub fn preprocess_encoded(&self, encoded: &[u8]) -> Result<PixelShufflePreprocessedImage> {
        self.preprocess_image(&image::load_from_memory(encoded)?.to_rgb8(), None)
    }

    pub fn preprocess_encoded_with_patch_limit(
        &self,
        encoded: &[u8],
        max_patches: usize,
    ) -> Result<PixelShufflePreprocessedImage> {
        self.preprocess_image(&image::load_from_memory(encoded)?.to_rgb8(), Some(max_patches))
    }

    pub fn preprocess_rgb(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
    ) -> Result<PixelShufflePreprocessedImage> {
        let image = RgbImage::from_raw(u32::try_from(width)?, u32::try_from(height)?, rgb.to_vec())
            .ok_or_else(|| invalid("RGB byte length does not match image dimensions"))?;
        self.preprocess_image(&image, None)
    }

    fn preprocess_image(
        &self,
        image: &RgbImage,
        runtime_max_patches: Option<usize>,
    ) -> Result<PixelShufflePreprocessedImage> {
        self.validate()?;
        let budget = self.patch_budget();
        let budget = runtime_max_patches.map_or(budget, |limit| limit.min(budget));
        let factor = self.shuffle_factor;
        if budget < factor * factor {
            return Err(invalid("pixel-shuffle vision runtime patch budget is too small"));
        }
        let (rows, cols) = target::patches(
            usize::try_from(image.height())?,
            usize::try_from(image.width())?,
            self,
            budget,
        )?;
        let scaled = self.rescaled(image)?;
        let (height, width) = (rows * self.patch_size, cols * self.patch_size);
        let resized = if (scaled.height, scaled.width) == (height, width) {
            scaled
        } else {
            scaled.resize(width, height)?
        };
        Ok(PixelShufflePreprocessedImage {
            patches: self.patchify(&resized, rows, cols)?,
            grid_height: rows,
            grid_width: cols,
            soft_tokens: (rows / factor) * (cols / factor),
        })
    }

    /// The image in floating point, rescaled before resizing so no value is
    /// rounded back to a byte.
    fn rescaled(&self, image: &RgbImage) -> Result<Planar> {
        let scale = numeric::narrow(self.rescale_factor)?;
        Ok(Planar {
            width: usize::try_from(image.width())?,
            height: usize::try_from(image.height())?,
            pixels: image.as_raw().iter().map(|value| f32::from(*value) * scale).collect(),
        })
    }

    fn patchify(&self, image: &Planar, rows: usize, cols: usize) -> Result<Vec<f32>> {
        let size = self.patch_size;
        let mut patches = Vec::with_capacity(rows * cols * 3 * size * size);
        for row in 0..rows {
            for col in 0..cols {
                for channel in 0..3 {
                    let mean = numeric::narrow(self.image_mean[channel])?;
                    let std = numeric::narrow(self.image_std[channel])?;
                    for y in 0..size {
                        let line = (row * size + y) * image.width + col * size;
                        for x in 0..size {
                            patches.push((image.pixels[(line + x) * 3 + channel] - mean) / std);
                        }
                    }
                }
            }
        }
        Ok(patches)
    }

    fn validate(&self) -> Result<()> {
        if self.patch_size == 0
            || self.shuffle_factor == 0
            || self.min_patches == 0
            || self.max_patches < self.min_patches
            || self.image_std.iter().any(|value| !value.is_finite() || *value == 0.0)
        {
            return Err(invalid("unsupported pixel-shuffle vision image processor configuration"));
        }
        Ok(())
    }
}

fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}
