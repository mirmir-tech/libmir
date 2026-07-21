use models::{layout::DecoderConfig, weights::WeightBindingPlan};

use super::{
    config::ClampedRoutedConfig,
    layer::ClampedRoutedLayer,
    projection::{BoundEmbedding, BoundLinear},
};
use crate::engine::{
    Array, DecoderCache, KvPageFormat, ModelTensors, NormWeight, Result, Stream, decode_graph,
    decoder::{LayerContext, LayerLoopOptions, forward_layers},
    lowering::{FeedForwardLowering, LayerLowering, MixerLowering},
};

#[derive(Debug)]
pub struct ClampedRoutedModel {
    layers: Vec<ClampedRoutedLayer>,
    windows: Vec<Option<usize>>,
    cache_step: usize,
    config: ClampedRoutedConfig,
    embedding: BoundEmbedding,
    final_norm: NormWeight,
    output: BoundLinear,
}

impl ClampedRoutedModel {
    pub fn load(
        tensors: &ModelTensors,
        decoder: &DecoderConfig,
        bindings: &WeightBindingPlan,
        lowering: &[LayerLowering],
        cache_step: usize,
        stream: &Stream,
    ) -> Result<Self> {
        let config = ClampedRoutedConfig::from_decoder(decoder)?;
        let compatible = lowering.len() == decoder.num_hidden_layers
            && lowering.iter().all(|layer| {
                layer.feed_forward == FeedForwardLowering::ClampedRouted
                    && matches!(layer.mixer, MixerLowering::Softmax { sinks: true, .. })
            });
        if !compatible {
            return Err(crate::engine::Error::InvalidModel(
                "clamped-routed loader requires attention sinks on every layer".into(),
            ));
        }
        let layers = (0..decoder.num_hidden_layers)
            .map(|index| {
                ClampedRoutedLayer::load(
                    tensors,
                    bindings.routed_decoder_layer(index)?,
                    config,
                    stream,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let boundary = bindings.decoder_boundary()?;
        let windows = lowering
            .iter()
            .map(|layer| match layer.mixer {
                MixerLowering::Softmax { window, .. } => window,
                MixerLowering::Linear => unreachable!("validated sink-attention mixer"),
            })
            .collect();
        Ok(Self {
            layers,
            windows,
            cache_step,
            config,
            embedding: BoundEmbedding::load_binding(tensors, boundary.embedding)?,
            final_norm: NormWeight::load_name(tensors, &boundary.final_norm.source)?,
            output: BoundLinear::load_binding(tensors, boundary.output, stream)?,
        })
    }

    pub fn new_cache(&self, stream: &Stream) -> Result<DecoderCache> {
        DecoderCache::new_with_format(
            &self.windows,
            self.cache_step,
            KvPageFormat::resolve(stream.config().kv_cache.dtype)?,
            stream.config().kv_cache.block_size,
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
            LayerLoopOptions::disabled(),
        )?;
        let hidden = self.final_norm.apply(&hidden, self.config.epsilon, stream)?;
        let logits = self.output.forward(&hidden, stream)?;
        if !causal {
            decode_graph::export_once(&logits, stream)?;
        }
        Ok(logits)
    }

    pub fn forward_packed_decode(
        &self,
        token_ids: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        let mut rows = Vec::with_capacity(caches.len());
        for (row, (cache, position)) in caches.iter_mut().zip(positions).enumerate() {
            let token = token_ids.slice(&[row, 0], &[row + 1, 1], stream)?;
            rows.push(self.forward(&token, cache, *position, false, stream)?);
        }
        let refs = rows.iter().collect::<Vec<_>>();
        Array::concatenate(&refs, 0, stream)
    }
}
