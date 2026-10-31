//! Rotary position embedding applied by the QKV postprocess kernels.

use super::super::geometry::narrow;
use crate::Result;

/// Rotary embedding over the leading `dimensions` of each query and key head.
/// Frequencies are spread over `pairing` dimensions; dimensions beyond
/// `dimensions` pass through unrotated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotaryEmbedding {
    pub dimensions: usize,
    pub pairing: usize,
    pub theta: f32,
}

impl RotaryEmbedding {
    pub(crate) fn fits(self, head_dim: usize) -> bool {
        self.dimensions != 0
            && self.dimensions <= self.pairing
            && self.pairing <= head_dim
            && self.dimensions.is_multiple_of(2)
            && self.pairing.is_multiple_of(2)
            && self.theta.is_finite()
            && self.theta > 0.0
    }

    /// Kernel arguments `(rotary_dim, pairing_dim, theta)`. Heads without a
    /// rotary embedding rotate zero dimensions and pass through.
    pub(crate) fn arguments(rotary: Option<Self>) -> Result<(u32, u32, f32)> {
        rotary.map_or(Ok((0, 0, 1.0)), |rotary| {
            Ok((narrow(rotary.dimensions)?, narrow(rotary.pairing)?, rotary.theta))
        })
    }
}
