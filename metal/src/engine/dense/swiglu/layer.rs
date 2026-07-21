use super::{attention, config::DenseSwiGluLayerConfig, weights::DenseSwiGluWeights};
use crate::engine::{
    Array, DecoderCache, Error, FusedAttention, FusedGateUp, KvCache, ModelTensors, Result, Stream,
    decoder::{LayerContext, LoweredLayer, LoweredPackedLayer, MixerKind},
    fusion_planner::{FusionPlanner, ProjectionBiases},
    lowering::LayerLowering,
};

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
        self.feed_forward(input, context.stream)
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
        self.feed_forward_packed(input, batch_size, stream)
    }
}

impl DenseSwiGluLayer {
    pub(super) fn load(
        tensors: &ModelTensors,
        bindings: DenseDecoderLayerBindings<'_>,
        lowering: LayerLowering,
        config: DenseSwiGluLayerConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let weights = DenseSwiGluWeights::load_bindings(tensors, bindings, config, stream)?;
        let fusion = FusionPlanner::new(stream).projections(
            lowering.feed_forward,
            ProjectionBiases::new(
                [weights.attention.query.has_bias(), weights.attention.key.has_bias()],
                Some(weights.attention.value.has_bias()),
                [weights.mlp.gate.has_bias(), weights.mlp.up.has_bias()],
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
        let fused_gate_up = fusion
            .gate_up
            .then(|| weights.mlp.gate.fuse_gate_up(&weights.mlp.up, stream))
            .transpose()?
            .flatten();
        fused_attention.as_ref().map_or(Ok(()), FusedAttention::warm)?;
        fused_gate_up.as_ref().map_or(Ok(()), FusedGateUp::warm)?;
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
        let normalized = self.weights.input_norm.apply(input, self.config.rms_norm_eps, stream)?;
        let attention = attention::forward(
            &normalized,
            &self.weights.attention,
            self.fused_attention.as_ref(),
            self.config,
            attention::AttentionContext { cache, position, causal, stream },
        )?;
        input.add(&attention, stream)
    }

    fn feed_forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let normalized =
            self.weights
                .post_attention_norm
                .apply(input, self.config.rms_norm_eps, stream)?;
        let fused = (input.shape()?.get(1) == Some(&1))
            .then_some(self.fused_gate_up.as_ref())
            .flatten();
        let (gate, up) = fused.map_or_else(
            || {
                Ok((
                    self.weights.mlp.gate.forward(&normalized, stream)?,
                    self.weights.mlp.up.forward(&normalized, stream)?,
                ))
            },
            |fused| fused.forward_pair(&normalized, stream),
        )?;
        let activated = gate.silu_mul(&up, stream)?;
        let feed_forward = self.weights.mlp.down.forward(&activated, stream)?;
        input.add(&feed_forward, stream)
    }

    pub(super) fn fusion_summary(&self) -> (bool, bool) {
        (self.fused_attention.is_some(), self.fused_gate_up.is_some())
    }
}
use models::weights::DenseDecoderLayerBindings;
