//! The payload-free kinds a layer is made of, as runtimes tell them apart.

use models::semantic::{AttentionWindow, QkNormalization, RouterNormalization};

use crate::set::{Flag, flag_enum, product_flag};

flag_enum! {
    /// How a layer mixes tokens. Softmax attention splits by the two
    /// properties a runtime's attention kernels either have or lack.
    pub enum MixerKind {
        Softmax,
        /// Softmax attention over a sliding window.
        SoftmaxSliding,
        /// Softmax attention with learned sink logits.
        SoftmaxSinks,
        SoftmaxSlidingSinks,
        Linear,
        StateSpace,
        Latent,
    }
}

impl MixerKind {
    /// The four softmax kinds.
    pub const SOFTMAX: [Self; 4] =
        [Self::Softmax, Self::SoftmaxSliding, Self::SoftmaxSinks, Self::SoftmaxSlidingSinks];
    /// Softmax attention without sinks.
    pub const SOFTMAX_WITHOUT_SINKS: [Self; 2] = [Self::Softmax, Self::SoftmaxSliding];
    /// Softmax attention with sinks.
    pub const SOFTMAX_WITH_SINKS: [Self; 2] = [Self::SoftmaxSinks, Self::SoftmaxSlidingSinks];

    /// The softmax kind of an attention with these properties; a chunked
    /// window is a decoder feature, not a sliding one.
    #[must_use]
    pub const fn softmax(sinks: bool, window: Option<AttentionWindow>) -> Self {
        let sliding = matches!(window, Some(AttentionWindow::Sliding(_)));
        match (sliding, sinks) {
            (false, false) => Self::Softmax,
            (true, false) => Self::SoftmaxSliding,
            (false, true) => Self::SoftmaxSinks,
            (true, true) => Self::SoftmaxSlidingSinks,
        }
    }
}

flag_enum! {
    /// The block structure of a layer's feed-forward.
    pub enum FeedForwardShape {
        Dense,
        /// Routed experts alone.
        Routed,
        /// Routed experts beside a shared expert.
        SharedRouted,
        DenseAndRouted,
        /// Routed experts in a projected latent space.
        LatentRouted,
        SharedLatentRouted,
        None,
    }
}

flag_enum! {
    /// A resolved feed-forward activation.
    pub enum ActivationKind {
        SwiGlu,
        ClampedSwiGlu,
        GeluTanh,
        ReluSquared,
    }
}

flag_enum! {
    /// Where in a feed-forward an activation runs.
    pub enum ActivationSite {
        /// A dense MLP's.
        Dense,
        /// Each routed expert's.
        Expert,
        /// A shared expert's.
        Shared,
    }
}

flag_enum! {
    /// A router's selection and weighting rule, without its parameters.
    pub enum RouterKind {
        SoftmaxTopK,
        SoftmaxProbabilityTopK,
        SigmoidBiasedTopK,
        SparseMixer,
        SigmoidInputScaledTopK,
    }
}

impl RouterKind {
    /// The routers that select the top-k experts by softmax or by biased
    /// sigmoid, and weigh their outputs.
    pub const TOP_K: [Self; 3] =
        [Self::SoftmaxTopK, Self::SoftmaxProbabilityTopK, Self::SigmoidBiasedTopK];

    /// Exhaustive: a new router fails to compile here, never passes silently.
    #[must_use]
    pub const fn of(router: &RouterNormalization) -> Self {
        match router {
            RouterNormalization::SoftmaxTopK => Self::SoftmaxTopK,
            RouterNormalization::SoftmaxProbabilityTopK => Self::SoftmaxProbabilityTopK,
            RouterNormalization::SigmoidBiasedTopK => Self::SigmoidBiasedTopK,
            RouterNormalization::SparseMixer { .. } => Self::SparseMixer,
            RouterNormalization::SigmoidInputScaledTopK => Self::SigmoidInputScaledTopK,
        }
    }
}

impl Flag for QkNormalization {
    const ALL: &'static [Self] = &[
        Self::None,
        Self::QueryKeyRms,
        Self::QueryKeyUnitRms,
        Self::QueryKeyRmsAcrossHeads,
        Self::QueryKeyValueRms,
    ];

    fn index(self) -> u32 {
        match self {
            Self::None => 0,
            Self::QueryKeyRms => 1,
            Self::QueryKeyUnitRms => 2,
            Self::QueryKeyRmsAcrossHeads => 3,
            Self::QueryKeyValueRms => 4,
        }
    }
}

/// A layer's mixer and feed-forward block structure together.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LayerShape {
    pub mixer: MixerKind,
    pub feed_forward: FeedForwardShape,
}

/// An activation at its site.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ActivationUse {
    pub site: ActivationSite,
    pub kind: ActivationKind,
}

/// A router of one feed-forward shape's experts.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RouterUse {
    pub feed_forward: FeedForwardShape,
    pub router: RouterKind,
}

product_flag!(LayerShape {
    mixer: MixerKind,
    feed_forward: FeedForwardShape
});
product_flag!(ActivationUse {
    site: ActivationSite,
    kind: ActivationKind
});
product_flag!(RouterUse {
    feed_forward: FeedForwardShape,
    router: RouterKind
});
