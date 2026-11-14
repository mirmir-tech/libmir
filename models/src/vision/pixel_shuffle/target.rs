//! The patch grid an image is resized to: its own aspect ratio, scaled
//! down to the budget or up to the minimum, each side a multiple of the
//! shuffle factor. Follows the processor's published rule step by step,
//! including Python's round-half-to-even. When that rule overshoots the
//! budget (its minimum-size upscale and per-axis rounding do not check the
//! product), both sides are fitted jointly to the budget instead.

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
    if budget >= config.min_patches && rows * cols < config.min_patches {
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
    if rows * cols <= budget {
        return Ok((rows, cols));
    }
    fit((closest_rows, closest_cols), factor, budget)
}

/// The largest factor-aligned grid near the image's aspect ratio whose
/// product stays within `budget`. Works in shuffle units (`factor` patches
/// per side): one axis takes the floor or ceiling of its budget-scaled
/// length, the other its aspect-matched length capped by what the budget
/// leaves, and the candidate with the most patches wins.
fn fit(closest: (f64, f64), factor: usize, budget: usize) -> Result<(usize, usize)> {
    let units = budget / (factor * factor);
    if units == 0 {
        return Err(invalid("pixel-shuffle vision runtime patch budget is too small"));
    }
    let limit = real(units)?;
    let aspect = (closest.0 / real(factor)?, closest.1 / real(factor)?);
    let scale = (limit / (aspect.0 * aspect.1)).sqrt();
    let mut best = (1, 1);
    for (transposed, fixed, other) in [(false, aspect.0, aspect.1), (true, aspect.1, aspect.0)] {
        let length = scale * fixed;
        for fixed_units in [length.floor(), length.ceil()] {
            let fixed_units = whole(fixed_units.clamp(1.0, limit))?;
            let matched = whole((other * real(fixed_units)? / fixed).round().max(1.0))?;
            let other_units = matched.min(units / fixed_units);
            let candidate = if transposed {
                (other_units, fixed_units)
            } else {
                (fixed_units, other_units)
            };
            if candidate.0 * candidate.1 > best.0 * best.1 {
                best = candidate;
            }
        }
    }
    Ok((best.0 * factor, best.1 * factor))
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
