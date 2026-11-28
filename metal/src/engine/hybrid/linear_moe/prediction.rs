//! Qwen3.5's multi-token prediction module on Metal.
//!
//! For row `i` the module reads the decoder's normalized output `h_i` and the
//! following token `t_{i+1}`:
//! `norm(layer(fc([enorm(embed(t_{i+1})) | hnorm(h_i)])))`, at position `i`.
//! The layer is a full-attention decoder layer with its own K/V cache; the
//! decoder's embedding and output head serve the module.

use architecture::{CacheLayout, LayerCacheEntry};
use models::{layout::DecoderConfig, semantic::DecoderSpec, weights::WeightBindingPlan};

use super::{HybridLinearMoeModel, layer::HybridLinearMoeLayer};
use crate::engine::{
    Array, DecoderCache, Error, ModelTensors, NormWeight, Result, Stream,
    binding::{BoundLinear, adjusted_norm},
    decoder::{LayerContext, LayerLoopOptions, forward_layers},
};

#[derive(Debug)]
pub struct PredictionModule {
    embedding_norm: NormWeight,
    hidden_norm: NormWeight,
    /// `[hidden, 2 · hidden]` over `[embedding | hidden]`.
    fusion: BoundLinear,
    layer: HybridLinearMoeLayer,
    final_norm: NormWeight,
    cache_entry: LayerCacheEntry,
}

impl PredictionModule {
    /// The checkpoint's first module, when speculation asks for it.
    pub(super) fn load(
        tensors: &ModelTensors,
        decoder: &DecoderConfig,
        spec: &DecoderSpec,
        cache_layout: &CacheLayout,
        bindings: &WeightBindingPlan,
        norm_shift: f32,
        stream: &Stream,
    ) -> Result<Option<Self>> {
        if stream.config().speculation().is_none() || !bindings.has_prediction_module(0) {
            return Ok(None);
        }
        let module = bindings.mixed_prediction_layer(0)?;
        let layer = spec.prediction_layers.first().ok_or_else(|| {
            Error::InvalidModel("the prediction module has no specification".into())
        })?;
        let cache_entry = *cache_layout.prediction_layers.first().ok_or_else(|| {
            Error::InvalidModel("the prediction module has no cache layout entry".into())
        })?;
        let norm = |binding| adjusted_norm(tensors, binding, norm_shift, stream);
        Ok(Some(Self {
            embedding_norm: norm(module.embedding_norm)?,
            hidden_norm: norm(module.hidden_norm)?,
            fusion: BoundLinear::load(tensors, module.fusion, stream)?,
            // Index 0 of the module's own one-layer cache.
            layer: HybridLinearMoeLayer::load(
                tensors, decoder, layer, 0, module.layer, norm_shift, stream,
            )?,
            final_norm: norm(module.final_norm)?,
            cache_entry,
        }))
    }
}

impl HybridLinearMoeModel {
    /// The prediction module's K/V, when it is loaded.
    pub(crate) fn new_prediction_cache(&self, stream: &Stream) -> Result<Option<DecoderCache>> {
        self.prediction
            .as_ref()
            .map(|module| {
                DecoderCache::from_entries(
                    std::slice::from_ref(&module.cache_entry),
                    self.cache_step,
                    crate::engine::KvPageFormat::resolve(stream.config().kv_cache.dtype)?,
                    stream.config().kv_cache.block_size,
                    stream.config().kv_cache.block_count as usize,
                    stream.paged_arenas(),
                )
            })
            .transpose()
    }

    /// The module's normalized rows `[1, L, hidden]`, whose logits predict
    /// the tokens after `next_tokens [1, L]`, from normalized rows
    /// `hidden [1, L, hidden]` (the decoder's, or the module's own when it
    /// chains drafts) at positions from `position`.
    pub(crate) fn forward_prediction(
        &self,
        hidden: &Array,
        next_tokens: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        let module = self
            .prediction
            .as_ref()
            .ok_or_else(|| Error::InvalidModel("the prediction module is not loaded".into()))?;
        let embedded = self.embedding.lookup(next_tokens, stream)?;
        let embedded = module.embedding_norm.apply(&embedded, self.rms_norm_eps, stream)?;
        let hidden = module.hidden_norm.apply(hidden, self.rms_norm_eps, stream)?;
        let fused = module
            .fusion
            .forward(&Array::concatenate(&[&embedded, &hidden], -1, stream)?, stream)?;
        let rows = fused.shape()?.get(1).copied().unwrap_or(1);
        let output = forward_layers(
            std::slice::from_ref(&module.layer),
            fused,
            cache,
            LayerContext {
                position,
                causal: rows > 1,
                positions: None,
                image: None,
                stream,
            },
            LayerLoopOptions::new(false, None, false),
        )?;
        module.final_norm.apply(&output, self.rms_norm_eps, stream)
    }
}
