//! User-facing names of what a runtime lacks, for refusal messages: the
//! typed values stay with the error, these words go to logs and the UI.

use std::fmt;

use models::semantic::QkNormalization;

use super::{Rejection, Runtime, TaskRefusal};
use crate::{
    capability::{
        ActivationKind, ActivationSite, AttentionScaleValue, EmbeddingScale, FeedForwardShape,
        MixerKind, Requirement, RouterKind, ScaleSource,
    },
    cuda::CudaDecoderRuntime,
    decoder::Feature,
    metal::MetalDecoderRuntime,
};

impl fmt::Display for Rejection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "the {} runtime lacks ", self.runtime)?;
        for (index, requirement) in self.missing.iter().enumerate() {
            if index > 0 {
                formatter.write_str(", ")?;
            }
            write!(formatter, "{requirement}")?;
        }
        Ok(())
    }
}

impl fmt::Display for Runtime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(runtime) => write!(formatter, "CUDA {}", cuda(*runtime)),
            Self::Metal(runtime) => write!(formatter, "Metal {}", metal(*runtime)),
        }
    }
}

const fn cuda(runtime: CudaDecoderRuntime) -> &'static str {
    match runtime {
        CudaDecoderRuntime::Dense => "dense",
        CudaDecoderRuntime::DenseAndRouted => "dense-and-routed",
        CudaDecoderRuntime::SharedRouted => "shared-routed",
        CudaDecoderRuntime::DenseMixed => "dense mixed-attention",
        CudaDecoderRuntime::ClampedRouted => "clamped-routed",
        CudaDecoderRuntime::StateSpaceHybrid => "state-space hybrid",
        CudaDecoderRuntime::LatentAttention => "latent-attention",
    }
}

const fn metal(runtime: MetalDecoderRuntime) -> &'static str {
    match runtime {
        MetalDecoderRuntime::Dense => "dense",
        MetalDecoderRuntime::DenseGelu => "dense GELU",
        MetalDecoderRuntime::DenseAndRouted => "dense-and-routed",
        MetalDecoderRuntime::SharedRouted => "shared-routed",
        MetalDecoderRuntime::DenseMixed => "dense mixed-attention",
        MetalDecoderRuntime::ClampedRouted => "clamped-routed",
        MetalDecoderRuntime::StateSpaceHybrid => "state-space hybrid",
        MetalDecoderRuntime::LatentAttention => "latent-attention",
    }
}

impl fmt::Display for Requirement {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Layer(layer) => write!(
                formatter,
                "{} layers with {}",
                mixer(layer.mixer),
                feed_forward(layer.feed_forward)
            ),
            Self::Activation(activation) => write!(
                formatter,
                "{} in {}",
                activation_kind(activation.kind),
                activation_site(activation.site)
            ),
            Self::Router(router) => write!(
                formatter,
                "{} routing for {}",
                router_kind(router.router),
                feed_forward(router.feed_forward)
            ),
            Self::Normalization(normalization) => formatter.write_str(qk(*normalization)),
            Self::Feature(feature) => formatter.write_str(self::feature(*feature)),
            Self::EmbeddingScale(scale) => formatter.write_str(match scale {
                EmbeddingScale::Unit => "unscaled embeddings",
                EmbeddingScale::SqrtHidden => "embeddings scaled by sqrt(hidden size)",
                EmbeddingScale::Other => "embeddings scaled by another factor",
            }),
            Self::AttentionScale(scale) => {
                formatter.write_str(match scale.value {
                    AttentionScaleValue::InverseSqrtHeadDim => "the default attention scale",
                    AttentionScaleValue::Unit => "a unit attention scale",
                    AttentionScaleValue::Other => "another attention scale",
                })?;
                if scale.source == ScaleSource::Override {
                    formatter.write_str(" overriding the configuration")?;
                }
                Ok(())
            },
        }
    }
}

const fn mixer(kind: MixerKind) -> &'static str {
    match kind {
        MixerKind::Softmax => "softmax attention",
        MixerKind::SoftmaxSliding => "sliding-window attention",
        MixerKind::SoftmaxSinks => "attention-sink",
        MixerKind::SoftmaxSlidingSinks => "sliding-window attention-sink",
        MixerKind::Linear => "linear attention",
        MixerKind::StateSpace => "state-space",
        MixerKind::Latent => "latent attention",
    }
}

const fn feed_forward(shape: FeedForwardShape) -> &'static str {
    match shape {
        FeedForwardShape::Dense => "a dense MLP",
        FeedForwardShape::Routed => "routed experts",
        FeedForwardShape::SharedRouted => "routed and shared experts",
        FeedForwardShape::DenseAndRouted => "a dense MLP beside routed experts",
        FeedForwardShape::LatentRouted => "latent routed experts",
        FeedForwardShape::SharedLatentRouted => "latent routed and shared experts",
        FeedForwardShape::None => "no feed-forward",
    }
}

const fn activation_kind(kind: ActivationKind) -> &'static str {
    match kind {
        ActivationKind::SwiGlu => "SwiGLU",
        ActivationKind::ClampedSwiGlu => "clamped SwiGLU",
        ActivationKind::GeluTanh => "tanh GELU",
        ActivationKind::ReluSquared => "relu²",
    }
}

const fn activation_site(site: ActivationSite) -> &'static str {
    match site {
        ActivationSite::Dense => "the dense MLP",
        ActivationSite::Expert => "routed experts",
        ActivationSite::Shared => "the shared expert",
    }
}

const fn router_kind(kind: RouterKind) -> &'static str {
    match kind {
        RouterKind::SoftmaxTopK => "softmax top-k",
        RouterKind::SoftmaxProbabilityTopK => "softmax-probability top-k",
        RouterKind::SigmoidBiasedTopK => "biased sigmoid top-k",
        RouterKind::SparseMixer => "SparseMixer",
        RouterKind::SigmoidInputScaledTopK => "input-scaled sigmoid top-k",
    }
}

const fn qk(normalization: QkNormalization) -> &'static str {
    match normalization {
        QkNormalization::None => "attention without Q/K norms",
        QkNormalization::QueryKeyRms => "Q/K RMS norms",
        QkNormalization::QueryKeyUnitRms => "unit Q/K RMS norms",
        QkNormalization::QueryKeyRmsAcrossHeads => "Q/K RMS norms across heads",
        QkNormalization::QueryKeyValueRms => "Q/K/V RMS norms",
    }
}

const fn feature(feature: Feature) -> &'static str {
    match feature {
        Feature::ResidualScaling => "residual or logit scaling",
        Feature::LayerNorm => "LayerNorm",
        Feature::PostOrParallelNorm => "post-norm or parallel blocks",
        Feature::NormWeightOffset => "norm weight offsets",
        Feature::ScoreSoftcap => "attention score soft-capping",
        Feature::ChunkedAttention => "chunked attention",
        Feature::QueryTemperature => "query temperature",
        Feature::LongRope => "LongRoPE",
        Feature::SharedKeyValue => "keys shared as values",
        Feature::RotaryAttention => "rotary attention",
        Feature::ExpertGroups => "grouped expert selection",
        Feature::ExpertOutputScale => "an expert output scale",
        Feature::AttentionFree => "decoders without attention",
        Feature::StateSpaceFree => "decoders without state-space layers",
        Feature::PerLayerInput => "per-layer inputs",
        Feature::SharedLayerKeyValues => "layers sharing keys and values",
        Feature::UnbiasedAttention => "attention projections without biases",
        Feature::NoRepeatedLayer => {
            "layers other than one YaRN-rotated attention layer with routed experts repeated"
        },
    }
}

impl fmt::Display for TaskRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::EmbeddingPooling => "embeddings are pooled only from the normalized last token",
            Self::SequenceScoringEncoder => {
                "sequence scoring runs only a packed-QKV LayerNorm GELU encoder with one label \
                 and supported RoPE"
            },
        })
    }
}
