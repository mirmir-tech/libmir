//! The pixel-shuffle vision encoder of Nemotron Omni checkpoints.
//!
//! Learned prefix tokens and patches with an absolute position table resized
//! to each image, pre-norm transformer blocks without rotary positions, then a
//! pixel shuffle and an RMS-normalized squared-rectifier projector into the
//! decoder's hidden size.

mod block;
mod embedding;
mod prefill;
mod projector;
mod tower;

#[cfg(test)]
mod tests;

pub use tower::PixelShuffleVisionTower;

use crate::engine::{Error, Result};

pub(super) fn dimension(value: usize, label: &str) -> Result<i32> {
    i32::try_from(value).map_or_else(
        |_| Err(Error::InvalidModel(format!("pixel-shuffle vision {label} exceeds i32"))),
        Ok,
    )
}

fn encoder(suffix: &str) -> String {
    models::weights::pixel_shuffle_encoder_tensor(suffix)
}
