use models::{semantic::DecoderSpec, weights::WeightBindingPlan};

use super::{
    config::ClampedRoutedConfig,
    layer::ClampedRoutedLayer,
    projection::{BoundEmbedding, BoundLinear},
};
use crate::engine::{
    Array, DecoderCache, KvPageFormat, ModelTensors, NormWeight, Result, Stream, decode,
    decoder::{LayerContext, LayerLoopOptions, forward_layers},
    lowering::{Activation, Experts, FeedForwardLowering, LayerLowering, MixerLowering},
};

#[derive(Debug)]
pub struct ClampedRoutedModel {
    pub(super) layers: Vec<ClampedRoutedLayer>,
    windows: Vec<Option<usize>>,
    cache_step: usize,
    pub(super) final_norm_eps: f32,
    pub(super) embedding: BoundEmbedding,
    pub(super) final_norm: NormWeight,
    pub(super) output: BoundLinear,
}

impl ClampedRoutedModel {
    #[cfg(test)]
    pub(crate) fn project_prefill_output(&self, hidden: &Array, stream: &Stream) -> Result<Array> {
        let hidden = self.final_norm.apply(hidden, self.final_norm_eps, stream)?;
        self.output.forward(&hidden, stream)
    }

    pub fn load(
        tensors: &ModelTensors,
        spec: &DecoderSpec,
        bindings: &WeightBindingPlan,
        lowering: &[LayerLowering],
        cache_step: usize,
        stream: &Stream,
    ) -> Result<Self> {
        let compatible = lowering.len() == spec.layers.len()
            && lowering.iter().all(|layer| {
                matches!(
                    layer.feed_forward,
                    FeedForwardLowering::Routed {
                        experts: Experts {
                            activation: Activation::SwiGlu { clamp: Some(_) },
                            ..
                        },
                        shared: None,
                    }
                ) && matches!(layer.mixer, MixerLowering::Softmax { sinks: true, .. })
            });
        if !compatible {
            return Err(crate::engine::Error::InvalidModel(
                "clamped-routed loader requires attention sinks on every layer".into(),
            ));
        }
        let layers = spec
            .layers
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                ClampedRoutedLayer::load(
                    tensors,
                    bindings.routed_decoder_layer(index)?,
                    layer,
                    ClampedRoutedConfig::from_spec(spec, layer)?,
                    stream,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let boundary = bindings.decoder_boundary()?;
        let windows = lowering
            .iter()
            .map(|layer| match layer.mixer {
                MixerLowering::Softmax { window, .. } => {
                    window.and_then(models::semantic::AttentionWindow::sliding)
                },
                MixerLowering::Linear | MixerLowering::StateSpace | MixerLowering::Latent(_) => {
                    unreachable!("validated sink-attention mixer")
                },
            })
            .collect();
        Ok(Self {
            layers,
            windows,
            cache_step,
            final_norm_eps: spec.final_norm.epsilon.to_string().parse()?,
            embedding: BoundEmbedding::load(tensors, boundary.embedding, stream)?,
            final_norm: NormWeight::load_name(tensors, &boundary.final_norm.source)?,
            output: BoundLinear::load(tensors, boundary.output, stream)?,
        })
    }

    pub fn new_cache(&self, stream: &Stream) -> Result<DecoderCache> {
        DecoderCache::new_with_pool_capacity(
            &self.windows,
            self.cache_step,
            KvPageFormat::resolve(stream.config().kv_cache.dtype)?,
            stream.config().kv_cache.block_size,
            DecoderCache::physical_page_capacity(stream, self.cache_step),
            stream.paged_arenas(),
        )
    }

    pub fn forward(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let hidden = self.forward_state(token_ids, cache, position, causal, stream)?;
        let hidden = self.final_norm.apply(&hidden, self.final_norm_eps, stream)?;
        let logits = self.output.forward(&hidden, stream)?;
        if !causal {
            decode::graph::export_once(&logits, stream)?;
        }
        Ok(logits)
    }

    pub fn forward_state(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let hidden = self.embedding.lookup(token_ids, stream)?;
        let hidden = forward_layers(
            &self.layers,
            hidden,
            cache,
            LayerContext {
                position,
                causal,
                positions: None,
                image: None,
                stream,
            },
            LayerLoopOptions::new(
                stream.config().diagnostics.profile_layers,
                None,
                stream.config().diagnostics.profile_graph_build,
            ),
        )?;
        Ok(hidden)
    }
}
