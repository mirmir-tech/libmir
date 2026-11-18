use models::{layout::DecoderConfig, weights::WeightBindingPlan};

use super::{
    config::DenseSwiGluLayerConfig,
    layer::DenseSwiGluLayer,
    projection::{BoundEmbedding, BoundLinear},
};
use crate::engine::{
    Array, DecoderCache, ModelTensors, NormWeight, Result, Stream, decode,
    decoder::{LayerContext, LayerLoopOptions, forward_layers},
    lowering::{FeedForwardLowering, LayerLowering, MixerLowering, NormalizationLowering},
};

#[derive(Debug)]
pub struct DenseSwiGluModel {
    pub(super) layers: Vec<DenseSwiGluLayer>,
    cache_step: usize,
    /// Factor on token embeddings before the first layer (Gemma's
    /// sqrt(hidden)).
    embedding_multiplier: Option<f32>,
    pub(super) embedding: BoundEmbedding,
    pub(super) output_projection: OutputProjection,
    pub(super) final_norm: NormWeight,
}

#[derive(Debug)]
pub(super) enum OutputProjection {
    TiedEmbedding,
    Linear(BoundLinear),
}

impl DenseSwiGluModel {
    pub fn load(
        tensors: &ModelTensors,
        decoder: &DecoderConfig,
        bindings: &WeightBindingPlan,
        lowering: &[LayerLowering],
        final_norm: NormalizationLowering,
        cache_step: usize,
        stream: &Stream,
    ) -> Result<Self> {
        let compatible = lowering.len() == decoder.num_hidden_layers
            && lowering.iter().all(|layer| {
                matches!(
                    layer.feed_forward,
                    FeedForwardLowering::Dense | FeedForwardLowering::Routed
                ) && matches!(layer.mixer, MixerLowering::Softmax { .. })
            });
        if decoder.final_logit_softcapping.is_some() {
            return Err(crate::engine::Error::InvalidModel(
                "dense Metal decoder has no final logit softcap".into(),
            ));
        }
        if !compatible {
            return Err(crate::engine::Error::InvalidModel(
                "dense SwiGLU loader requires softmax attention with gated or routed MLPs".into(),
            ));
        }
        let mut layers = Vec::with_capacity(decoder.num_hidden_layers);
        for (index, lowered) in lowering.iter().enumerate() {
            let config = DenseSwiGluLayerConfig::from_decoder(decoder, index)?;
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
            OutputProjection::Linear(BoundLinear::load(tensors, boundary.output, stream)?)
        };
        let embedding_multiplier = decoder
            .embedding_multiplier
            .filter(|multiplier| multiplier.to_bits() != 1.0_f64.to_bits())
            .map(|multiplier| multiplier.to_string().parse::<f32>())
            .transpose()?;
        Ok(Self {
            layers,
            cache_step,
            embedding_multiplier,
            embedding: BoundEmbedding::load(tensors, boundary.embedding, stream)?,
            output_projection,
            final_norm: NormWeight::load_name_adjusted(
                tensors,
                &boundary.final_norm.source,
                super::config::norm_shift(final_norm),
                stream,
            )?,
        })
    }

    pub fn new_cache(&self, stream: &Stream) -> Result<DecoderCache> {
        DecoderCache::new_with_pool_capacity(
            &self.layers.iter().map(|layer| layer.config.window).collect::<Vec<_>>(),
            self.cache_step,
            crate::engine::KvPageFormat::resolve(stream.config().kv_cache.dtype)?,
            stream.config().kv_cache.block_size,
            DecoderCache::physical_page_capacity(stream, self.cache_step),
            stream.paged_arenas(),
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
        decode::graph::export_once(&logits, stream)?;
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

    pub fn forward_prefill_state(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        self.forward_layers(token_ids, cache, position, true, stream)
    }

    fn forward_hidden(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let hidden = self.forward_layers(token_ids, cache, position, causal, stream)?;
        let hidden = self.final_norm.apply(&hidden, 1.0e-6, stream)?;
        match &self.output_projection {
            OutputProjection::TiedEmbedding => self.embedding.project(&hidden, stream),
            OutputProjection::Linear(output_head) => output_head.forward(&hidden, stream),
        }
    }

    fn forward_layers(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let hidden = self.embed(token_ids, stream)?;
        let shape = token_ids.shape()?;
        let batch = usize::try_from(shape[0])?;
        let sequence = usize::try_from(shape[1])?;
        let position_tokens = usize::try_from(position)?;
        let evaluation_step = causal
            .then(|| {
                crate::engine::decoder::prefill_evaluation_step(
                    batch,
                    sequence,
                    position_tokens,
                    self.layers.len(),
                )
            })
            .flatten();
        forward_layers(
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
            LayerLoopOptions::new(false, evaluation_step, false),
        )
    }

    /// Token embeddings, scaled by the declared multiplier.
    pub(super) fn embed(&self, token_ids: &Array, stream: &Stream) -> Result<Array> {
        let hidden = self.embedding.lookup(token_ids, stream)?;
        match self.embedding_multiplier {
            Some(multiplier) => hidden.multiply_scalar(multiplier, stream),
            None => Ok(hidden),
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
