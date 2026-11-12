//! Antialiased bicubic resizing in floating point, without clamping: the
//! encoder was trained on pixels resized this way, so ringing past 0 and 1
//! at sharp edges is part of its input. Separable, horizontal then
//! vertical, with the filter widened by the scale when shrinking.

use super::numeric::{narrow, real, whole};
use crate::error::Result;

/// Cubic convolution coefficient of the antialiased filter.
const A: f64 = -0.5;
/// Support of the cubic filter at unit scale.
const SUPPORT: f64 = 2.0;

/// An RGB image of `width` × `height` with interleaved channels.
pub(super) struct Planar {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<f32>,
}

impl Planar {
    pub(super) fn resize(&self, width: usize, height: usize) -> Result<Self> {
        let horizontal = weights(self.width, width)?;
        let mut wide = vec![0.0f32; height.max(self.height) * width * 3];
        for y in 0..self.height {
            for (x, (start, taps)) in horizontal.iter().enumerate() {
                for channel in 0..3 {
                    wide[(y * width + x) * 3 + channel] =
                        (taps.iter().enumerate()).fold(0.0, |sum, (k, tap)| {
                            tap.mul_add(
                                self.pixels[(y * self.width + start + k) * 3 + channel],
                                sum,
                            )
                        });
                }
            }
        }
        let vertical = weights(self.height, height)?;
        let mut pixels = vec![0.0f32; height * width * 3];
        for (y, (start, taps)) in vertical.iter().enumerate() {
            for x in 0..width {
                for channel in 0..3 {
                    pixels[(y * width + x) * 3 + channel] =
                        (taps.iter().enumerate()).fold(0.0, |sum, (k, tap)| {
                            tap.mul_add(wide[((start + k) * width + x) * 3 + channel], sum)
                        });
                }
            }
        }
        Ok(Self { width, height, pixels })
    }
}

/// For each output index, the first input index and the normalized taps.
fn weights(input: usize, output: usize) -> Result<Vec<(usize, Vec<f32>)>> {
    let scale = real(input)? / real(output)?;
    let support = SUPPORT * scale.max(1.0);
    let inverse = if scale > 1.0 {
        scale.recip()
    } else {
        1.0
    };
    (0..output)
        .map(|index| {
            let center = scale * (real(index)? + 0.5);
            // The window as the reference kernel truncates it.
            let start = whole((center - support + 0.5).max(0.0).floor())?;
            let end = whole((center + support + 0.5).floor())?.min(input);
            let taps = (start..end)
                .map(|j| Ok(cubic((real(j)? - center + 0.5) * inverse)))
                .collect::<Result<Vec<f64>>>()?;
            let total: f64 = taps.iter().sum();
            let taps = taps
                .into_iter()
                .map(|tap| {
                    narrow(if total == 0.0 {
                        tap
                    } else {
                        tap / total
                    })
                })
                .collect::<Result<Vec<f32>>>()?;
            Ok((start, taps))
        })
        .collect()
}

fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0).mul_add(x, -(A + 3.0)) * x).mul_add(x, 1.0)
    } else if x < 2.0 {
        (x - 5.0).mul_add(x, 8.0).mul_add(x, -4.0) * A
    } else {
        0.0
    }
}
