use models::{layout::DecoderConfig, weights::WeightBindingPlan};

use super::{config::DenseSwiGluLayerConfig, layer::DenseSwiGluLayer};
use crate::engine::{
    Array, DecoderCache, ModelTensors, NormWeight, QuantizedEmbedding, QuantizedLinear, Result,
    Stream,
    binding::{affine_embedding, affine_linear},
    decode_graph,
    decoder::{LayerContext, LayerLoopOptions, forward_layers},
    lowering::{FeedForwardLowering, LayerLowering, MixerLowering},
};

#[derive(Debug)]
pub struct DenseSwiGluModel {
    pub(super) layers: Vec<DenseSwiGluLayer>,
    cache_step: usize,
    pub(super) embedding: QuantizedEmbedding,
    pub(super) output_projection: OutputProjection,
    pub(super) final_norm: NormWeight,
}

#[derive(Debug)]
pub(super) enum OutputProjection {
    TiedEmbedding,
    Linear(QuantizedLinear),
}

impl DenseSwiGluModel {
    pub fn load(
        tensors: &ModelTensors,
        decoder: &DecoderConfig,
        bindings: &WeightBindingPlan,
        lowering: &[LayerLowering],
        group_size: usize,
        cache_step: usize,
        stream: &Stream,
    ) -> Result<Self> {
        let compatible = lowering.len() == decoder.num_hidden_layers
            && lowering.iter().all(|layer| {
                layer.feed_forward == FeedForwardLowering::Dense
                    && matches!(layer.mixer, MixerLowering::Softmax { window: None, .. })
            });
        if !compatible {
            return Err(crate::engine::Error::InvalidModel(
                "dense SwiGLU loader requires non-windowed softmax and dense layers".into(),
            ));
        }
        let mut layers = Vec::with_capacity(decoder.num_hidden_layers);
        for (index, lowered) in lowering.iter().enumerate() {
            let config = DenseSwiGluLayerConfig::from_decoder(decoder, group_size)?;
            layers.push(DenseSwiGluLayer::load(
                tensors,
                bindings.dense_decoder_layer(index)?,
                *lowered,
                config,
                stream,
            )?);
        }
        let boundary = bindings.decoder_boundary_with_tied_output(decoder.tie_word_embeddings)?;
        let output_projection = if decoder.tie_word_embeddings {
            OutputProjection::TiedEmbedding
        } else {
            OutputProjection::Linear(affine_linear(tensors, boundary.output)?)
        };
        Ok(Self {
            layers,
            cache_step,
            embedding: affine_embedding(tensors, boundary.embedding)?,
            output_projection,
            final_norm: NormWeight::load_name(tensors, &boundary.final_norm.source)?,
        })
    }

    pub fn new_cache(&self, stream: &Stream) -> Result<DecoderCache> {
        DecoderCache::new_with_format(
            &vec![None; self.layers.len()],
            self.cache_step,
            crate::engine::KvPageFormat::resolve(stream.config().kv_cache.dtype)?,
            stream.config().kv_cache.block_size,
        )
    }

    pub fn forward_decode(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        let logits = self.forward_hidden(token_ids, cache, position, false, stream)?;
        decode_graph::export_once(&logits, stream)?;
        Ok(logits)
    }

    pub fn forward_prefill(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        self.forward_hidden(token_ids, cache, position, true, stream)
    }

    fn forward_hidden(
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
        let hidden = self.final_norm.apply(&hidden, 1.0e-6, stream)?;
        match &self.output_projection {
            OutputProjection::TiedEmbedding => self.embedding.project(&hidden, stream),
            OutputProjection::Linear(output_head) => output_head.forward(&hidden, stream),
        }
    }

    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    #[must_use]
    pub fn fusion_summary(&self) -> (usize, usize) {
        self.layers.iter().fold((0, 0), |counts, layer| {
            let (attention, gate_up) = layer.fusion_summary();
            (counts.0 + usize::from(attention), counts.1 + usize::from(gate_up))
        })
    }
}
