//! How a unit router weighs the experts it selects.

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
}

/// Weighting of the selected experts and a factor on every weight, which
/// scales the combined routed output.
#[derive(Clone, Copy, Debug)]
pub struct RouterUnitScoring {
    pub weighting: RouterUnitWeighting,
    pub scale: f32,
}

impl RouterUnitScoring {
    pub const RENORMALIZED: Self = Self {
        weighting: RouterUnitWeighting::Renormalized,
        scale: 1.0,
    };
}

// The scale is a fixed configuration constant; plans compare it exactly.
impl PartialEq for RouterUnitScoring {
    fn eq(&self, other: &Self) -> bool {
        self.weighting == other.weighting && self.scale.to_bits() == other.scale.to_bits()
    }
}

impl Eq for RouterUnitScoring {}

impl std::hash::Hash for RouterUnitScoring {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.weighting.hash(state);
        self.scale.to_bits().hash(state);
    }
}
