//! Why a backend refuses a model's architecture, typed so that callers and
//! tests can tell what is missing; `Display` is for logs.

mod display;

use std::fmt;

use crate::{capability::Requirement, cuda::CudaDecoderRuntime, metal::MetalDecoderRuntime};

pub type Result<T> = std::result::Result<T, ArchitectureError>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArchitectureError {
    /// A generation task reached admission without its semantic spec.
    MissingSemanticContract,
    /// A layer (or the whole decoder) declares what no kernel of the backend
    /// computes, whichever runtime runs it.
    Unlowerable {
        layer: Option<usize>,
        feature: Unlowerable,
    },
    /// No runtime executes the lowered plan; each refusal names what that
    /// runtime lacks.
    NoRuntime { rejections: Vec<Rejection> },
    /// The task itself, before any decoder.
    Task(TaskRefusal),
}

/// What a backend's lowering cannot express.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Unlowerable {
    /// An activation the semantic spec kept only by name.
    Activation { name: String },
    /// A latent attention geometry the latent kernels do not take.
    LatentGeometry {
        key_value_rank: usize,
        rope_head_dim: usize,
    },
    /// Latent keys not rotated, or rotated other than plainly, linearly, by
    /// `YaRN` or `LongRoPE`.
    LatentRotary,
    /// Latent attention with no rotary key part to rotate.
    LatentUnrotated,
    /// A latent query temperature counting positions from other than zero.
    LatentQueryTemperatureOffset { offset: usize },
    /// Latent routed experts without a projected latent space.
    LatentExpertsWithoutLatentSize,
    /// A clamped `SwiGLU` slope other than the one the kernel computes.
    ClampAlpha { alpha: Constant },
    /// More experts than the router kernel ranks.
    ExpertCount { experts: usize, max: usize },
    /// More expert groups than the router kernel ranks.
    ExpertGroups { groups: usize, max: usize },
    /// A norm weight offset other than none or Gemma's unit offset.
    NormWeightOffset { offset: Constant },
    /// A `LayerNorm` with a weight offset.
    LayerNormWeightOffset,
    /// A dimension beyond what the backend's kernels index.
    Dimension { what: &'static str, value: usize },
}

/// A model constant quoted by a refusal, compared by its bits so that the
/// refusal stays `Eq`.
#[derive(Clone, Copy, Debug)]
pub struct Constant(pub f64);

impl PartialEq for Constant {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for Constant {}

impl fmt::Display for Constant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// A task the backend does not run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaskRefusal {
    /// Embedding that does not pool the normalized last token.
    EmbeddingPooling,
    /// A sequence-scoring encoder other than packed-QKV `LayerNorm` GELU
    /// with one label and supported `RoPE`.
    SequenceScoringEncoder,
}

/// One runtime's refusal of a plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rejection {
    pub runtime: Runtime,
    pub missing: Vec<Requirement>,
}

/// A decoder runtime of either backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Runtime {
    Cuda(CudaDecoderRuntime),
    Metal(MetalDecoderRuntime),
}

impl fmt::Display for ArchitectureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSemanticContract => {
                formatter.write_str("the generation semantic contract is missing")
            },
            Self::Unlowerable { layer: Some(layer), feature } => {
                write!(formatter, "layer {layer}: {feature}")
            },
            Self::Unlowerable { layer: None, feature } => write!(formatter, "{feature}"),
            Self::NoRuntime { rejections } => {
                formatter.write_str("no runtime executes the decoder")?;
                for rejection in rejections {
                    write!(formatter, "; {rejection}")?;
                }
                Ok(())
            },
            Self::Task(refusal) => write!(formatter, "{refusal}"),
        }
    }
}

impl fmt::Display for Unlowerable {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Activation { name } => write!(formatter, "no kernel runs the {name} activation"),
            Self::LatentGeometry { key_value_rank, rope_head_dim } => write!(
                formatter,
                "no latent attention kernel for a {key_value_rank}-wide latent with a {rope_head_dim}-wide rotary key"
            ),
            Self::LatentRotary => formatter
                .write_str("latent keys must rotate plainly, linearly, by YaRN or by LongRoPE"),
            Self::LatentUnrotated => {
                formatter.write_str("latent attention needs a rotary key part")
            },
            Self::LatentQueryTemperatureOffset { offset } => {
                write!(formatter, "latent queries temper positions from zero, not from {offset}")
            },
            Self::LatentExpertsWithoutLatentSize => formatter
                .write_str("latent routed experts need a projected latent space (moe_latent_size)"),
            Self::ClampAlpha { alpha } => {
                write!(formatter, "no clamped SwiGLU kernel computes a slope of {alpha}")
            },
            Self::ExpertCount { experts, max } => {
                write!(formatter, "{experts} experts exceed the router's {max}")
            },
            Self::ExpertGroups { groups, max } => {
                write!(formatter, "{groups} expert groups exceed the router's {max}")
            },
            Self::NormWeightOffset { offset } => {
                write!(formatter, "no norm kernel adds a weight offset of {offset}")
            },
            Self::LayerNormWeightOffset => {
                formatter.write_str("no LayerNorm kernel adds a weight offset")
            },
            Self::Dimension { what, value } => write!(formatter, "{what} {value} is too large"),
        }
    }
}

impl std::error::Error for ArchitectureError {}
