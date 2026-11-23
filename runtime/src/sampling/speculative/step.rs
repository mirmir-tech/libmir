//! The target of a device-sampled speculative step.

use super::DraftSampling;
use crate::{
    backend::SamplingLogits,
    error::{Result, RuntimeError},
};

/// A device-sampled decode step: its target distribution, the host's draw
/// and the vocabulary bound of its logits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampledStep {
    pub sampling: DraftSampling,
    pub draw: f32,
    pub vocab_size: usize,
}

impl SampledStep {
    /// The sampled step `sampling` asks for; `None` for greedy, masked and
    /// host-sampled steps.
    #[must_use]
    pub const fn of(sampling: &SamplingLogits) -> Option<Self> {
        match *sampling {
            SamplingLogits::Sample {
                vocab_size,
                temperature,
                top_p,
                top_k,
                draw,
            } => Some(Self {
                sampling: DraftSampling { temperature, top_p, top_k },
                draw,
                vocab_size,
            }),
            SamplingLogits::SampleTopK { k, vocab_size, temperature, draw } => Some(Self {
                sampling: DraftSampling { temperature, top_p: 1.0, top_k: k },
                draw,
                vocab_size,
            }),
            SamplingLogits::None
            | SamplingLogits::Masked { .. }
            | SamplingLogits::Full
            | SamplingLogits::TopK { .. } => None,
        }
    }

    /// The rejection-sampling seed of the step at `position`.
    pub fn seed(self, position: usize) -> Result<u64> {
        speculative_seed(self.draw, position)
    }
}

/// The rejection-sampling seed of a step at `position` with the host's
/// `draw`: the draw's bits under the position.
pub fn speculative_seed(draw: f32, position: usize) -> Result<u64> {
    let position = u64::try_from(position)
        .map_err(|_| RuntimeError::Backend("speculative position overflows u64".into()))?;
    Ok(u64::from(draw.to_bits()) | (position << 32))
}
