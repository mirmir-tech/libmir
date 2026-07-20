mod load;
mod plan;
mod session;
#[cfg(all(test, target_os = "linux"))]
mod tests;

use models::layout::DecoderConfig;
use runtime::kv::{CacheConfig, KvStorageSpec};

use self::load::{build_layer, infer_norm_shift, required_norm};
pub use self::session::CudaHybridLinearModelSession;
use crate::{
    AffineQuantizedEmbedding, AffineQuantizedWeight, CudaAffineGatedDeltaMoeLayer,
    CudaAffineGatedFullAttentionMoeLayer, CudaAffineGatedFullAttentionState, CudaAffineOutputHead,
    CudaBackend, CudaGatedDeltaState, CudaTensor, CudaTensorSet, Error, Result,
};

#[derive(Clone, Debug)]
enum HybridLayerTemplate {
    Linear(Box<CudaAffineGatedDeltaMoeLayer>),
    Full(Box<CudaAffineGatedFullAttentionMoeLayer>),
}

#[derive(Debug)]
pub enum CudaHybridLinearLayerState {
    Linear(CudaGatedDeltaState),
    Full(Box<CudaAffineGatedFullAttentionState>),
}

/// Structurally assembled affine hybrid linear/full-attention model weights.
#[derive(Clone, Debug)]
pub struct CudaHybridLinearModelTemplate {
    backend: CudaBackend,
    decoder: DecoderConfig,
    embedding: AffineQuantizedWeight,
    final_norm: CudaTensor,
    output: AffineQuantizedWeight,
    layers: Vec<HybridLayerTemplate>,
    cache: CacheConfig,
    max_sequence_blocks: usize,
    norm_shift: f32,
}

impl CudaHybridLinearModelTemplate {
    pub fn from_tensors(
        backend: &CudaBackend,
        decoder: &DecoderConfig,
        tensors: &CudaTensorSet,
        cache: CacheConfig,
        max_sequence_blocks: usize,
    ) -> Result<Self> {
        if !decoder.uses_hybrid_linear_moe_stack() || max_sequence_blocks == 0 {
            return Err(Error::UnsupportedDecoderLayer(
                "parsed decoder is not a hybrid linear/full-attention MoE stack".into(),
            ));
        }
        let embedding = AffineQuantizedWeight::load(tensors, "language_model.model.embed_tokens")?;
        embedding.infer_config(1, decoder.hidden_size, decoder.vocab_size)?;
        let output = if decoder.tie_word_embeddings {
            embedding.clone()
        } else {
            AffineQuantizedWeight::load(tensors, "language_model.lm_head")?
        };
        output.infer_config(1, decoder.hidden_size, decoder.vocab_size)?;
        let final_norm =
            required_norm(tensors, "language_model.model.norm.weight", decoder.hidden_size)?;
        let norm_shift = infer_norm_shift(tensors, decoder)?;
        let layers = (0..decoder.num_hidden_layers)
            .map(|layer| build_layer(backend, decoder, tensors, layer, norm_shift))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            backend: backend.clone(),
            decoder: decoder.clone(),
            embedding,
            final_norm,
            output,
            layers,
            cache,
            max_sequence_blocks,
            norm_shift,
        })
    }

    #[must_use]
    pub const fn decoder(&self) -> &DecoderConfig {
        &self.decoder
    }

    #[must_use]
    pub const fn norm_shift(&self) -> f32 {
        self.norm_shift
    }

    pub fn prepare_embedding(&self) -> Result<AffineQuantizedEmbedding> {
        let config =
            self.embedding
                .infer_config(1, self.decoder.hidden_size, self.decoder.vocab_size)?;
        self.backend.prepare_affine_embedding(config, 1.0)
    }

    pub fn prepare_output_head(&self) -> Result<CudaAffineOutputHead> {
        CudaAffineOutputHead::from_weight(
            &self.backend,
            self.decoder.hidden_size,
            self.decoder.vocab_size,
            &self.output,
        )
    }

    pub fn prepare_states(&self) -> Result<Vec<CudaHybridLinearLayerState>> {
        self.layers
            .iter()
            .enumerate()
            .map(|(index, layer)| match layer {
                HybridLayerTemplate::Linear(layer) => {
                    layer.prepare_state().map(CudaHybridLinearLayerState::Linear)
                },
                HybridLayerTemplate::Full(layer) => {
                    let storage = KvStorageSpec::new(
                        self.cache,
                        self.decoder.layer_key_value_heads(index),
                        self.decoder.layer_head_dim(index),
                    );
                    layer
                        .prepare_state(index, storage, self.max_sequence_blocks)
                        .map(Box::new)
                        .map(CudaHybridLinearLayerState::Full)
                },
            })
            .collect()
    }

    pub fn instantiate(&self) -> Result<CudaHybridLinearModelSession> {
        CudaHybridLinearModelSession::new(self)
    }

    #[must_use]
    pub const fn embedding_weight(&self) -> &AffineQuantizedWeight {
        &self.embedding
    }

    #[must_use]
    pub const fn final_norm_weight(&self) -> &CudaTensor {
        &self.final_norm
    }
}
