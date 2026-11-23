use models::{layout::DecoderConfig, weights::WeightBindingPlan};

use super::{
    config::DenseSwiGluLayerConfig,
    layer::{DenseSwiGluLayer, LayerBindings},
    projection::{BoundEmbedding, BoundLinear},
};
use crate::engine::{
    Array, DecoderCache, ModelTensors, NormWeight, Result, Stream, decode,
    decoder::{LayerContext, LayerLoopOptions, forward_layers},
    lowering::{LayerLowering, NormalizationLowering},
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
    /// `1 / logits_scaling` (Granite), when the decoder declares a divisor.
    logit_scale: Option<f32>,
    /// `cap · tanh(logits / cap)` (Gemma 2).
    logit_softcap: Option<f32>,
    pub(super) final_norm: NormWeight,
    pub(super) final_norm_eps: f32,
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
        super::config::require_dense_layers(decoder, lowering)?;
        let mut layers = Vec::with_capacity(decoder.num_hidden_layers);
        for (index, lowered) in lowering.iter().enumerate() {
            let config = DenseSwiGluLayerConfig::from_decoder(decoder, index)?;
            let layer_bindings = match lowered.norm_placement {
                models::semantic::NormPlacement::Pre => {
                    LayerBindings::Pre(bindings.dense_decoder_layer(index)?)
                },
                models::semantic::NormPlacement::Post => {
                    LayerBindings::Post(bindings.post_norm_decoder_layer(index)?)
                },
                models::semantic::NormPlacement::Parallel => {
                    LayerBindings::Parallel(bindings.parallel_decoder_layer(index)?)
                },
            };
            layers.push(DenseSwiGluLayer::load(
                tensors, decoder, layer_bindings, *lowered, config, stream,
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
        let logit_scale = decoder
            .logits_divisor
            .filter(|divisor| divisor.to_bits() != 1.0_f64.to_bits())
            .map(|divisor| divisor.recip().to_string().parse::<f32>())
            .transpose()?;
        Ok(Self {
            layers,
            cache_step,
            embedding_multiplier,
            logit_scale,
            logit_softcap: decoder
                .final_logit_softcapping
                .map(|cap| cap.to_string().parse::<f32>())
                .transpose()?,
            embedding: BoundEmbedding::load(tensors, boundary.embedding, stream)?,
            output_projection,
            final_norm: {
                let norm = NormWeight::load_name_adjusted(
                    tensors,
                    &boundary.final_norm.source,
                    super::config::norm_shift(final_norm),
                    stream,
                )?;
                match final_norm {
                    NormalizationLowering::Layer => norm.into_layer(
                        boundary
                            .final_norm
                            .dense_bias()
                            .map(|bias| tensors.get(bias))
                            .transpose()?,
                        stream,
                    )?,
                    NormalizationLowering::Rms | NormalizationLowering::RmsUnitOffset => norm,
                }
            },
            final_norm_eps: decoder.rms_norm_eps.to_string().parse()?,
        })
    }

    pub fn new_cache(&self, stream: &Stream) -> Result<DecoderCache> {
        DecoderCache::new_with_pool_capacity(
            &self.layers.iter().map(|layer| layer.config.span()).collect::<Vec<_>>(),
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
        let hidden = self.final_norm.apply(&hidden, self.final_norm_eps, stream)?;
        self.logits(&hidden, stream)
    }

    /// Logits of final-norm hidden states.
    pub(super) fn logits(&self, hidden: &Array, stream: &Stream) -> Result<Array> {
        let logits = match &self.output_projection {
            OutputProjection::TiedEmbedding => self.embedding.project(hidden, stream)?,
            OutputProjection::Linear(output_head) => output_head.forward(hidden, stream)?,
        };
        let logits = match self.logit_scale {
            Some(scale) => logits.multiply_scalar(scale, stream)?,
            None => logits,
        };
        match self.logit_softcap {
            Some(cap) => logits.logit_softcap(cap, stream),
            None => Ok(logits),
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
