use models::layout::DecoderConfig;

use super::{config::DenseSwiGluLayerConfig, layer::DenseSwiGluLayer};
use crate::engine::{
    Array, DecoderCache, ModelTensors, NormWeight, QuantizedEmbedding, QuantizedLinear, Result,
    Stream, decode_graph,
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
        group_size: usize,
        cache_step: usize,
        stream: &Stream,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(decoder.num_hidden_layers);
        for index in 0..decoder.num_hidden_layers {
            let config = DenseSwiGluLayerConfig::from_decoder(index, decoder, group_size)?;
            layers.push(DenseSwiGluLayer::load(tensors, config, stream)?);
        }
        let group_size = i32::try_from(group_size)?;
        let output_projection = if decoder.tie_word_embeddings {
            OutputProjection::TiedEmbedding
        } else {
            OutputProjection::Linear(QuantizedLinear::load(tensors, "lm_head", group_size)?)
        };
        Ok(Self {
            layers,
            cache_step,
            embedding: QuantizedEmbedding::load(tensors, "model.embed_tokens", group_size)?,
            output_projection,
            final_norm: NormWeight::load(tensors, "model.norm")?,
        })
    }

    pub fn new_cache(&self) -> Result<DecoderCache> {
        DecoderCache::new(&vec![None; self.layers.len()], self.cache_step)
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
        let mut hidden = self.embedding.lookup(token_ids, stream)?;
        for (layer, cache) in self.layers.iter().zip(cache.attention_caches_mut()?.iter_mut()) {
            hidden = layer.forward(&hidden, cache, position, causal, stream)?;
        }
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
