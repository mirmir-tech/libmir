//! Legalization against a target: a lowered plan states what it requires,
//! each runtime declares what it executes in the same terms, and one generic
//! check names what a runtime lacks.

mod kinds;
mod scale;

pub use kinds::{
    ActivationKind, ActivationSite, ActivationUse, FeedForwardShape, LayerShape, MixerKind,
    RouterKind, RouterUse,
};
use models::semantic::QkNormalization;
pub use scale::{AttentionScale, AttentionScaleValue, EmbeddingScale, ScaleSource};

use crate::{
    decoder::{DecoderFeatures, Feature},
    lowering::{FeedForwardLowering, MixerLowering},
    set::{Flag, Set},
};

/// Operations in the shared vocabulary: what a plan requires, or what a
/// runtime executes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Operations {
    pub layers: Set<LayerShape>,
    pub activations: Set<ActivationUse>,
    pub routers: Set<RouterUse>,
    pub normalizations: Set<QkNormalization>,
    pub features: Set<Feature>,
    pub embedding_scales: Set<EmbeddingScale>,
    pub attention_scales: Set<AttentionScale>,
}

/// One operation a runtime lacks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Requirement {
    Layer(LayerShape),
    Activation(ActivationUse),
    Router(RouterUse),
    Normalization(QkNormalization),
    Feature(Feature),
    EmbeddingScale(EmbeddingScale),
    AttentionScale(AttentionScale),
}

impl Operations {
    /// What a plan of these layers and decoder features requires.
    pub fn required<'a>(
        layers: impl IntoIterator<Item = (&'a MixerLowering, &'a FeedForwardLowering)>,
        features: DecoderFeatures,
    ) -> Self {
        let mut required = Self {
            features: features.flags,
            embedding_scales: Set::EMPTY.with(features.embedding),
            attention_scales: features.attention,
            ..Self::default()
        };
        for (mixer, feed_forward) in layers {
            let shape = feed_forward.shape();
            required.layers.insert(LayerShape { mixer: mixer.kind(), feed_forward: shape });
            for activation in feed_forward.activations() {
                required.activations.insert(activation);
            }
            if let Some(router) = feed_forward.router() {
                required.routers.insert(RouterUse { feed_forward: shape, router });
            }
            if let Some(normalization) = mixer.normalization() {
                required.normalizations.insert(normalization);
            }
        }
        required
    }

    /// What `self`, a runtime's capabilities, lacks of `required`.
    #[must_use]
    pub fn missing(&self, required: &Self) -> Vec<Requirement> {
        fn lacking<T: Flag>(
            has: Set<T>,
            needs: Set<T>,
            requirement: fn(T) -> Requirement,
        ) -> impl Iterator<Item = Requirement> {
            needs.difference(has).iter().map(requirement)
        }
        lacking(self.layers, required.layers, Requirement::Layer)
            .chain(lacking(self.activations, required.activations, Requirement::Activation))
            .chain(lacking(self.routers, required.routers, Requirement::Router))
            .chain(lacking(
                self.normalizations,
                required.normalizations,
                Requirement::Normalization,
            ))
            .chain(lacking(self.features, required.features, Requirement::Feature))
            .chain(lacking(
                self.embedding_scales,
                required.embedding_scales,
                Requirement::EmbeddingScale,
            ))
            .chain(lacking(
                self.attention_scales,
                required.attention_scales,
                Requirement::AttentionScale,
            ))
            .collect()
    }
}

/// Builders for capability tables.
impl Operations {
    /// Every layer of these mixers with these feed-forwards.
    #[must_use]
    pub fn layers(
        mut self,
        mixers: impl IntoIterator<Item = MixerKind>,
        feed_forwards: &[FeedForwardShape],
    ) -> Self {
        for mixer in mixers {
            for &feed_forward in feed_forwards {
                self.layers.insert(LayerShape { mixer, feed_forward });
            }
        }
        self
    }

    #[must_use]
    pub fn activations(mut self, site: ActivationSite, kinds: &[ActivationKind]) -> Self {
        for &kind in kinds {
            self.activations.insert(ActivationUse { site, kind });
        }
        self
    }

    /// These routers for the experts of each of these feed-forwards.
    #[must_use]
    pub fn routers(mut self, feed_forwards: &[FeedForwardShape], routers: &[RouterKind]) -> Self {
        for &feed_forward in feed_forwards {
            for &router in routers {
                self.routers.insert(RouterUse { feed_forward, router });
            }
        }
        self
    }

    #[must_use]
    pub fn normalizations(mut self, normalizations: &[QkNormalization]) -> Self {
        for &normalization in normalizations {
            self.normalizations.insert(normalization);
        }
        self
    }

    /// These embedding factors, and attention scales of these values from
    /// these sources.
    #[must_use]
    pub fn scales(
        mut self,
        embeddings: &[EmbeddingScale],
        values: &[AttentionScaleValue],
        sources: &[ScaleSource],
    ) -> Self {
        self.embedding_scales = embeddings.iter().copied().collect();
        for &value in values {
            for &source in sources {
                self.attention_scales.insert(AttentionScale { value, source });
            }
        }
        self
    }

    /// Every embedding factor and attention scale.
    #[must_use]
    pub fn any_scale(mut self) -> Self {
        self.embedding_scales = Set::all();
        self.attention_scales = Set::all();
        self
    }

    /// Every feature except `refused`.
    #[must_use]
    pub fn features_except(mut self, refused: &[Feature]) -> Self {
        self.features = Set::all().difference(refused.iter().copied().collect());
        self
    }
}
