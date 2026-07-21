mod load;
mod plan;
mod session;
#[cfg(all(test, target_os = "linux"))]
mod tests;

use models::{
    layout::DecoderConfig,
    semantic::{FeedForwardSpec, MixerSpec, SemanticModelSpec},
    weights::WeightBindingPlan,
};
use runtime::kv::{CacheConfig, KvStorageSpec};

use self::load::{build_layer, infer_norm_shift, required_norm};
pub use self::session::CudaSharedRoutedModelSession;
use crate::{
    AffineQuantizedEmbedding, AffineQuantizedWeight, CudaAffineGatedDeltaMoeLayer,
    CudaAffineGatedFullAttentionMoeLayer, CudaAffineGatedFullAttentionState, CudaAffineOutputHead,
    CudaBackend, CudaGatedDeltaState, CudaTensor, CudaTensorSet, Error, Result,
};

#[derive(Clone, Debug)]
enum SharedRoutedLayerTemplate {
    Linear(Box<CudaAffineGatedDeltaMoeLayer>),
    Full(Box<CudaAffineGatedFullAttentionMoeLayer>),
}

#[derive(Debug)]
pub enum CudaSharedRoutedLayerState {
    Linear(CudaGatedDeltaState),
    Full(Box<CudaAffineGatedFullAttentionState>),
}

/// Structurally assembled affine shared-routed model weights with per-layer
/// mixers.
#[derive(Clone, Debug)]
pub struct CudaSharedRoutedModelTemplate {
    backend: CudaBackend,
    decoder: DecoderConfig,
    embedding: AffineQuantizedWeight,
    final_norm: CudaTensor,
    output: AffineQuantizedWeight,
    layers: Vec<SharedRoutedLayerTemplate>,
    cache: CacheConfig,
    max_sequence_blocks: usize,
    norm_shift: f32,
}

impl CudaSharedRoutedModelTemplate {
    pub fn from_tensors(
        backend: &CudaBackend,
        decoder: &DecoderConfig,
        semantic: &SemanticModelSpec,
        tensors: &CudaTensorSet,
        bindings: &WeightBindingPlan,
        cache: CacheConfig,
        max_sequence_blocks: usize,
    ) -> Result<Self> {
        let compatible = semantic.decoder.layers.iter().all(|layer| {
            matches!(layer.feed_forward, FeedForwardSpec::Routed { shared: Some(_), .. })
        }) && semantic
            .decoder
            .layers
            .iter()
            .any(|layer| matches!(layer.mixer, MixerSpec::LinearAttention(_)))
            && semantic
                .decoder
                .layers
                .iter()
                .any(|layer| matches!(layer.mixer, MixerSpec::SoftmaxAttention(_)));
        if !compatible || max_sequence_blocks == 0 {
            return Err(Error::UnsupportedDecoderLayer(
                "parsed decoder is not a shared-routed mixed-mixer stack".into(),
            ));
        }
        let boundary = bindings.decoder_boundary()?;
        let embedding = AffineQuantizedWeight::load_binding(tensors, boundary.embedding)?;
        embedding.infer_config(1, decoder.hidden_size, decoder.vocab_size)?;
        let output = if decoder.tie_word_embeddings {
            embedding.clone()
        } else {
            AffineQuantizedWeight::load_binding(tensors, boundary.output)?
        };
        output.infer_config(1, decoder.hidden_size, decoder.vocab_size)?;
        let final_norm = required_norm(tensors, &boundary.final_norm.source, decoder.hidden_size)?;
        let norm_shift = infer_norm_shift(tensors, decoder, bindings)?;
        let layers = (0..decoder.num_hidden_layers)
            .map(|layer| {
                build_layer(
                    backend,
                    decoder,
                    tensors,
                    layer,
                    bindings.hybrid_decoder_layer(layer)?,
                    norm_shift,
                )
            })
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

    pub fn prepare_states(&self) -> Result<Vec<CudaSharedRoutedLayerState>> {
        self.layers
            .iter()
            .enumerate()
            .map(|(index, layer)| match layer {
                SharedRoutedLayerTemplate::Linear(layer) => {
                    layer.prepare_state().map(CudaSharedRoutedLayerState::Linear)
                },
                SharedRoutedLayerTemplate::Full(layer) => {
                    let storage = KvStorageSpec::new(
                        self.cache,
                        self.decoder.layer_key_value_heads(index),
                        self.decoder.layer_head_dim(index),
                    );
                    layer
                        .prepare_state(index, storage, self.max_sequence_blocks)
                        .map(Box::new)
                        .map(CudaSharedRoutedLayerState::Full)
                },
            })
            .collect()
    }

    pub fn instantiate(&self) -> Result<CudaSharedRoutedModelSession> {
        CudaSharedRoutedModelSession::new(self)
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
