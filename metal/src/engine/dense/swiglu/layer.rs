use std::time::Instant;

use super::{
    attention,
    config::DenseSwiGluLayerConfig,
    weights::{DenseSwiGluWeights, FeedForwardWeights},
};
use crate::engine::{
    Array, DecoderCache, Error, FusedAttention, FusedGateUp, KvCache, ModelTensors, Result, Stream,
    decoder::{LayerContext, LoweredLayer, LoweredPackedLayer, MixerKind},
    fusion_planner::{FusionPlanner, ProjectionBiases},
    gate_up_tuning,
    lowering::LayerLowering,
};

/// A layer's bindings by where it normalizes.
#[derive(Debug, Clone, Copy)]
pub(super) enum LayerBindings<'a> {
    Pre(models::weights::DenseDecoderLayerBindings<'a>),
    Post(models::weights::PostNormDecoderLayerBindings<'a>),
    Parallel(models::weights::ParallelDecoderLayerBindings<'a>),
}

#[derive(Debug)]
pub(super) struct DenseSwiGluLayer {
    pub(super) config: DenseSwiGluLayerConfig,
    pub(super) weights: DenseSwiGluWeights,
    pub(super) fused_attention: Option<FusedAttention>,
    pub(super) fused_gate_up: Option<FusedGateUp>,
}

impl LoweredLayer for DenseSwiGluLayer {
    fn mixer_kind(&self) -> MixerKind {
        MixerKind::Softmax
    }

    fn forward_mixer(
        &self,
        input: &Array,
        cache: &mut DecoderCache,
        index: usize,
        context: LayerContext<'_>,
    ) -> Result<Array> {
        let cache = cache
            .attention_caches_mut()?
            .get_mut(index)
            .ok_or_else(|| Error::InvalidModel(format!("missing cache for layer {index}")))?;
        self.mix(input, cache, context.position, context.causal, context.stream)
    }

    fn forward_feed_forward(&self, input: &Array, context: LayerContext<'_>) -> Result<Array> {
        // A parallel block's mixer already added the feed-forward.
        if self.weights.parallel() {
            return Array::from_native(input.native().clone());
        }
        self.feed_forward(input, context.causal, context.stream)
    }
}

impl LoweredPackedLayer for DenseSwiGluLayer {
    fn forward_packed_mixer(
        &self,
        input: &Array,
        caches: &mut [&mut DecoderCache],
        index: usize,
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        let mut layer_caches = caches
            .iter_mut()
            .map(|cache| {
                cache
                    .attention_caches_mut()?
                    .get_mut(index)
                    .ok_or_else(|| Error::InvalidModel(format!("missing cache for layer {index}")))
            })
            .collect::<Result<Vec<_>>>()?;
        self.mix_packed(input, &mut layer_caches, positions, stream)
    }

    fn forward_packed_feed_forward(
        &self,
        input: &Array,
        batch_size: usize,
        stream: &Stream,
    ) -> Result<Array> {
        if self.weights.parallel() {
            return Array::from_native(input.native().clone());
        }
        self.feed_forward_packed(input, batch_size, stream)
    }
}

impl DenseSwiGluLayer {
    pub(super) fn load(
        tensors: &ModelTensors,
        decoder: &models::layout::DecoderConfig,
        bindings: LayerBindings<'_>,
        lowering: LayerLowering,
        config: DenseSwiGluLayerConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let crate::engine::lowering::MixerLowering::Softmax { normalization, .. } = lowering.mixer
        else {
            return Err(Error::InvalidModel("dense layer requires softmax attention".into()));
        };
        if lowering.post_attention_norm != lowering.input_norm {
            return Err(Error::InvalidModel("dense layer norms scale differently".into()));
        }
        let config = DenseSwiGluLayerConfig { norm: lowering.input_norm, ..config };
        let weights = match bindings {
            LayerBindings::Pre(bindings) => DenseSwiGluWeights::load_bindings(
                tensors, decoder, bindings, config, normalization, stream,
            )?,
            LayerBindings::Post(bindings) => DenseSwiGluWeights::load_post_norm(
                tensors, decoder, bindings, config, normalization, stream,
            )?,
            LayerBindings::Parallel(bindings) => DenseSwiGluWeights::load_parallel(
                tensors, decoder, bindings, config, normalization, stream,
            )?,
        };
        let fusion = FusionPlanner::new(stream).projections(
            lowering.feed_forward,
            ProjectionBiases::new(
                [weights.attention.query.has_bias(), weights.attention.key.has_bias()],
                Some(weights.attention.value.has_bias()),
                match &weights.feed_forward {
                    FeedForwardWeights::Gated(mlp) => [mlp.gate.has_bias(), mlp.up.has_bias()],
                    FeedForwardWeights::Routed(_) | FeedForwardWeights::SharedRouted(_) => {
                        [false, false]
                    },
                },
            ),
        );
        let fused_attention = fusion
            .attention
            .then(|| {
                weights.attention.query.fuse_attention(
                    &weights.attention.key,
                    Some(&weights.attention.value),
                    stream,
                )
            })
            .transpose()?
            .flatten();
        let fused_gate_up = match &weights.feed_forward {
            FeedForwardWeights::Gated(mlp) if fusion.gate_up => {
                mlp.gate.fuse_gate_up(&mlp.up, stream)?
            },
            _ => None,
        };
        fused_attention.as_ref().map_or(Ok(()), |fused| fused.warm(stream))?;
        fused_gate_up.as_ref().map_or(Ok(()), |fused| fused.warm(stream))?;
        Ok(Self {
            config,
            weights,
            fused_attention,
            fused_gate_up,
        })
    }

    fn mix(
        &self,
        input: &Array,
        cache: &mut KvCache,
        position: i32,
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let started = profile_started(stream);
        let normalized = self.weights.attention_input(input, self.config.rms_norm_eps, stream)?;
        let attention = attention::forward(
            &normalized,
            &self.weights.attention,
            self.fused_attention.as_ref(),
            self.config,
            attention::AttentionContext { cache, position, causal, stream },
        )?;
        let output = self.attention_join(input, &attention, stream, |input| {
            self.feed_forward(input, causal, stream)
        })?;
        emit_profile(&output, stream, "attention", causal, started)?;
        Ok(output)
    }

    fn feed_forward(&self, input: &Array, causal: bool, stream: &Stream) -> Result<Array> {
        let started = profile_started(stream);
        let normalized =
            self.weights.feed_forward_input(input, self.config.rms_norm_eps, stream)?;
        let feed_forward = match &self.weights.feed_forward {
            FeedForwardWeights::Gated(mlp) => {
                let fused = (input.shape()?.get(1) == Some(&1))
                    .then_some(self.fused_gate_up.as_ref())
                    .flatten();
                let (gate, up) =
                    gate_up_tuning::forward(&mlp.gate, &mlp.up, fused, &normalized, stream)?;
                mlp.down.forward(&self.gated(&gate, &up, stream)?, stream)?
            },
            FeedForwardWeights::Routed(routed) => routed.forward(&normalized, stream)?,
            FeedForwardWeights::SharedRouted(experts) => experts.forward(&normalized, stream)?,
        };
        let output = self.feed_forward_residual(input, &feed_forward, stream)?;
        emit_profile(&output, stream, "feed_forward", causal, started)?;
        Ok(output)
    }

    pub(super) fn fusion_summary(&self) -> (bool, bool) {
        (self.fused_attention.is_some(), self.fused_gate_up.is_some())
    }
}

fn profile_started(stream: &Stream) -> Option<Instant> {
    stream.config().diagnostics.profile_components.then(Instant::now)
}

fn emit_profile(
    output: &Array,
    stream: &Stream,
    component: &str,
    causal: bool,
    started: Option<Instant>,
) -> Result<()> {
    let Some(started) = started else {
        return Ok(());
    };
    output.async_eval(stream)?;
    stream.synchronize()?;
    tracing::debug!(
        component,
        phase = if causal {
            "prefill"
        } else {
            "decode"
        },
        milliseconds = started.elapsed().as_secs_f64() * 1_000.0,
        "MLX dense component profile"
    );
    Ok(())
}
