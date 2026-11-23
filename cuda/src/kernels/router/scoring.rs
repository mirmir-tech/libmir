//! How a unit router weighs the experts it selects.

use models::semantic::{ExpertGroupLimit, RoutedExpertsSpec, RouterNormalization};

use crate::{Error, Result};

/// How the selected experts are chosen and weighted.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RouterUnitWeighting {
    /// Top-k logits, softmax over the selected ones.
    Renormalized,
    /// Top-k logits weighted by their softmax probability over every expert,
    /// not renormalized.
    Probability,
    /// Top-k of `sigmoid(logit) + correction_bias`, weighted by
    /// `sigmoid(logit)` renormalized over the selected ones; the router is
    /// compiled with the bias.
    SigmoidBiased,
    /// Llama 4: top-k logits weighted by their sigmoid, which scales each
    /// selected expert's input ([`super::RouterInputScale`]) instead of its
    /// output.
    SigmoidInputScaled,
    /// Phi-3.5-MoE's sparsemixer: each round takes the best remaining
    /// logit, weighted by its softmax over the remaining experts within
    /// `2 * jitter` of it, relative to `max(|logit|, best)`.
    SparseMixer(RouterJitter),
}

/// The sparsemixer's positive, finite `jitter_eps`, compared by its bits.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RouterJitter(u32);

impl RouterJitter {
    pub fn new(jitter: f32) -> Result<Self> {
        if jitter.is_finite() && jitter > 0.0 {
            Ok(Self(jitter.to_bits()))
        } else {
            Err(Error::InvalidRouter("sparsemixer jitter must be positive and finite"))
        }
    }

    #[must_use]
    pub const fn value(self) -> f32 {
        f32::from_bits(self.0)
    }
}

impl TryFrom<RouterNormalization> for RouterUnitWeighting {
    type Error = Error;

    fn try_from(normalization: RouterNormalization) -> Result<Self> {
        Ok(match normalization {
            RouterNormalization::SoftmaxTopK => Self::Renormalized,
            RouterNormalization::SoftmaxProbabilityTopK => Self::Probability,
            RouterNormalization::SigmoidBiasedTopK => Self::SigmoidBiased,
            RouterNormalization::SparseMixer { jitter_eps } =>
            {
                #[allow(clippy::cast_possible_truncation)]
                Self::SparseMixer(RouterJitter::new(jitter_eps as f32)?)
            },
            RouterNormalization::SigmoidInputScaledTopK => Self::SigmoidInputScaled,
            RouterNormalization::UnitTopK => {
                return Err(Error::UnsupportedDecoderLayer(format!(
                    "CUDA has no {normalization:?} unit router"
                )));
            },
        })
    }
}

/// Weighting of the selected experts, the groups they may come from, and a
/// factor on every weight, which scales the combined routed output.
#[derive(Clone, Copy, Debug)]
pub struct RouterUnitScoring {
    pub weighting: RouterUnitWeighting,
    /// Probability and renormalized routers rank a group by its best logit,
    /// the sigmoid router by its two best biased scores.
    pub group_limit: Option<ExpertGroupLimit>,
    pub scale: f32,
}

impl RouterUnitScoring {
    pub const RENORMALIZED: Self = Self {
        weighting: RouterUnitWeighting::Renormalized,
        group_limit: None,
        scale: 1.0,
    };

    /// The unit router of `routed`, with its output scale.
    pub fn of(routed: &RoutedExpertsSpec) -> Result<Self> {
        #[allow(clippy::cast_possible_truncation)]
        let scale = routed.output_scale as f32;
        Ok(Self {
            weighting: routed.router_normalization.try_into()?,
            group_limit: routed.group_limit,
            scale,
        })
    }
}

// The scale is a fixed configuration constant; plans compare it exactly.
impl PartialEq for RouterUnitScoring {
    fn eq(&self, other: &Self) -> bool {
        self.weighting == other.weighting
            && self.group_limit == other.group_limit
            && self.scale.to_bits() == other.scale.to_bits()
    }
}

impl Eq for RouterUnitScoring {}

impl std::hash::Hash for RouterUnitScoring {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.weighting.hash(state);
        self.group_limit.hash(state);
        self.scale.to_bits().hash(state);
    }
}
