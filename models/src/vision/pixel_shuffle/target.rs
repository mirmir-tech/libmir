//! The patch grid an image is resized to: its own aspect ratio, scaled
//! down to the budget or up to the minimum, each side a multiple of the
//! shuffle factor. Follows the processor's published rule step by step,
//! including Python's round-half-to-even.

use super::{
    invalid,
    numeric::{real, whole},
};
use crate::{error::Result, layout::PixelShuffleImageProcessorConfig};

/// `(rows, cols)` of patches for an image of `height` × `width` pixels.
pub(super) fn patches(
    height: usize,
    width: usize,
    config: &PixelShuffleImageProcessorConfig,
    budget: usize,
) -> Result<(usize, usize)> {
    if height == 0 || width == 0 {
        return Err(invalid("pixel-shuffle vision image is empty"));
    }
    let size = real(config.patch_size)?;
    let closest_rows = round_half_even(real(height)? / size + 0.5).max(1.0);
    let closest_cols = round_half_even(real(width)? / size + 0.5).max(1.0);
    let scale = (real(budget)? / (closest_rows * closest_cols)).sqrt().min(1.0);
    let mut rows = whole((scale * closest_rows).floor().max(1.0))?;
    let mut cols = whole((scale * closest_cols).floor().max(1.0))?;
    if budget > config.min_patches && rows * cols < config.min_patches {
        let up = (real(config.min_patches)? / real(rows * cols)?).sqrt();
        rows = whole((up * real(rows)?).ceil())?;
        cols = whole((up * real(cols)?).ceil())?;
    }
    let factor = config.shuffle_factor;
    if !rows.is_multiple_of(factor) {
        let up = rows + factor - rows % factor;
        rows = if up * cols <= budget {
            up
        } else {
            factor.max(rows - rows % factor)
        };
    }
    if !cols.is_multiple_of(factor) {
        let up = cols + factor - cols % factor;
        cols = if rows * up <= budget {
            up
        } else {
            factor.max(cols - cols % factor)
        };
    }
    Ok((rows, cols))
}

/// Python's `round`: halves go to the even neighbour.
fn round_half_even(value: f64) -> f64 {
    let floor = value.floor();
    let fraction = value - floor;
    let half = (fraction - 0.5).abs() < f64::EPSILON;
    if fraction > 0.5 || (half && floor % 2.0 != 0.0) {
        floor + 1.0
    } else {
        floor
    }
}

#[cfg(test)]
pub(super) fn round(value: f64) -> f64 {
    round_half_even(value)
}
