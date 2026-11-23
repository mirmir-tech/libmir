mod load;

use super::{config::DenseSwiGluLayerConfig, projection::BoundLinear, routed::RoutedMlp};
use crate::engine::{Array, NormWeight, Result, Stream};

#[derive(Debug)]
pub(super) struct AttentionWeights {
    pub(super) query: BoundLinear,
    pub(super) key: BoundLinear,
    pub(super) value: BoundLinear,
    pub(super) output: BoundLinear,
    pub(super) query_norm: Option<NormWeight>,
    pub(super) key_norm: Option<NormWeight>,
    pub(super) norm_scope: QkNormScope,
    pub(super) rope_frequencies: Option<Array>,
}

/// What a Q/K norm normalizes: each head, or the whole projection (`OLMo` 2);
/// `UnitHead` normalizes each head without a learned scale (Llama 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum QkNormScope {
    Head,
    Projection,
    UnitHead,
}

#[derive(Debug)]
pub(super) struct MlpWeights {
    pub(super) gate: BoundLinear,
    pub(super) up: BoundLinear,
    pub(super) down: BoundLinear,
}

/// The layer's feed-forward: a gated MLP, or routed experts in its place.
#[derive(Debug)]
pub(super) enum FeedForwardWeights {
    Gated(MlpWeights),
    Routed(RoutedMlp),
    /// Routed plus shared experts behind the decoder's router (Qwen2-MoE,
    /// GLM-4.5).
    SharedRouted(Box<crate::engine::SharedExpertMoe>),
}

/// Gemma-style norms on each sub-block's output before its residual add.
#[derive(Debug)]
pub(super) struct SandwichNorms {
    pub(super) attention_output: NormWeight,
    pub(super) feed_forward_output: NormWeight,
}

/// Where the layer normalizes: before each sub-block (with Gemma's output
/// norms when present), or only on each sub-block's output (`OLMo` 2).
#[derive(Debug)]
pub(super) enum LayerNorms {
    Pre {
        input: NormWeight,
        feed_forward: NormWeight,
        output: Option<SandwichNorms>,
    },
    Post(SandwichNorms),
    /// One norm feeds attention and the feed-forward, which read the same
    /// stream (Cohere).
    Parallel {
        input: NormWeight,
    },
}

#[derive(Debug)]
pub(super) struct DenseSwiGluWeights {
    pub(super) norms: LayerNorms,
    pub(super) attention: AttentionWeights,
    pub(super) feed_forward: FeedForwardWeights,
}

impl DenseSwiGluWeights {
    /// The attention input: the normalized stream, or the stream itself.
    pub(super) fn attention_input(
        &self,
        input: &Array,
        eps: f32,
        stream: &Stream,
    ) -> Result<Array> {
        match &self.norms {
            LayerNorms::Pre { input: norm, .. } | LayerNorms::Parallel { input: norm } => {
                norm.apply(input, eps, stream)
            },
            LayerNorms::Post(_) => Array::from_native(input.native().clone()),
        }
    }

    /// The feed-forward input: the normalized stream, or the stream itself.
    pub(super) fn feed_forward_input(
        &self,
        input: &Array,
        eps: f32,
        stream: &Stream,
    ) -> Result<Array> {
        match &self.norms {
            LayerNorms::Pre { feed_forward, .. } => feed_forward.apply(input, eps, stream),
            LayerNorms::Parallel { input: norm } => norm.apply(input, eps, stream),
            LayerNorms::Post(_) => Array::from_native(input.native().clone()),
        }
    }

    /// Norms on each sub-block's output before its residual add.
    pub(super) const fn output_norms(&self) -> Option<&SandwichNorms> {
        match &self.norms {
            LayerNorms::Pre { output, .. } => output.as_ref(),
            LayerNorms::Post(output) => Some(output),
            LayerNorms::Parallel { .. } => None,
        }
    }

    /// Whether attention and the feed-forward read the same stream.
    pub(super) const fn parallel(&self) -> bool {
        matches!(self.norms, LayerNorms::Parallel { .. })
    }
}

pub(super) fn rope_frequencies(
    config: DenseSwiGluLayerConfig,
    stream: &Stream,
) -> Result<Option<Array>> {
    config
        .rope_scaling
        .filter(|_| config.rope_dimensions > 0)
        .map(|scaling| match scaling {
            // Linear scaling divides positions: `rope_position_scale`.
            models::layout::RopeScaling::Linear { .. } => Ok(None),
            models::layout::RopeScaling::PiecewiseFrequency {
                factor,
                low_frequency_factor,
                high_frequency_factor,
                original_context_len,
            } => Array::piecewise_rope_frequencies(
                config.rope_dimensions,
                config.rope_base,
                factor.to_string().parse()?,
                low_frequency_factor.to_string().parse()?,
                high_frequency_factor.to_string().parse()?,
                i32::try_from(original_context_len)?,
                stream,
            )
            .map(Some),
            models::layout::RopeScaling::Yarn {
                factor,
                beta_fast,
                beta_slow,
                original_context_len,
                truncate,
                ..
            } => Array::yarn_rope_frequencies(
                config.rope_dimensions,
                config.rope_base,
                factor.to_string().parse()?,
                beta_fast.to_string().parse()?,
                beta_slow.to_string().parse()?,
                i32::try_from(original_context_len)?,
                truncate,
                stream,
            )
            .map(Some),
            // Each pair's period `base^(2i/d)` times its factor.
            models::layout::RopeScaling::LongRope { factors, .. } => {
                let width = f64::from(config.rope_dimensions);
                let base = f64::from(config.rope_base);
                let periods = factors
                    .as_slice()
                    .iter()
                    .enumerate()
                    .map(|(pair, &factor)| {
                        let exponent = 2.0 * f64::from(u32::try_from(pair)?) / width;
                        Ok(format!("{}", f64::from(factor) * base.powf(exponent)).parse()?)
                    })
                    .collect::<Result<Vec<f32>>>()?;
                if periods.len() * 2 != usize::try_from(config.rope_dimensions)? {
                    return Err(crate::engine::Error::InvalidModel(format!(
                        "LongRoPE declares {} factors for {} rotary dimensions",
                        periods.len(),
                        config.rope_dimensions
                    )));
                }
                Array::from_f32(&periods, &[i32::try_from(periods.len())?]).map(Some)
            },
        })
        .transpose()
        .map(Option::flatten)
}
