mod batch;
mod boundary;
mod checkpoint;
mod load;
mod memory;
mod plan;
mod position;
mod prediction;
mod session;
mod states;
#[cfg(all(test, target_os = "linux"))]
mod tests;

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use models::{
    layout::DecoderConfig,
    semantic::SemanticModelSpec,
    weights::{TensorCatalog, WeightBindingPlan},
};
use runtime::{kv::CacheConfig, speculation::Speculation};

pub(in crate::backend) use self::batch::DecodeWorkspaceKey;
pub use self::{
    batch::{CudaSharedRoutedDecodeBatch, CudaSharedRoutedPrefillBatch, DraftingRow},
    checkpoint::SharedRoutedCheckpoint,
    session::{CudaSharedRoutedModelSession, DraftRow, SpeculativeStep},
};
use self::{
    load::{build_layer, infer_norm_shift, required_norm},
    plan::SharedRoutedExecutionPlanCache,
    prediction::PredictionTemplate,
};
use crate::{
    CudaAffineGatedDeltaMoeLayer, CudaAffineGatedFullAttentionMoeLayer,
    CudaAffineGatedFullAttentionState, CudaAttentionHybridLayer, CudaBackend, CudaGatedDeltaState,
    CudaMamba2HybridLayer, CudaMamba2State, CudaTensor, CudaTensorSet, DecodeAttentionBf16,
    PagedKvCache, Result, backend::linear::CheckpointProjectionWeight,
};

#[derive(Clone, Debug)]
enum SharedRoutedLayerTemplate {
    Linear(Box<CudaAffineGatedDeltaMoeLayer>),
    Full(Box<CudaAffineGatedFullAttentionMoeLayer>),
    Mamba2(Box<CudaMamba2HybridLayer>),
    Attention(Box<CudaAttentionHybridLayer>),
}

#[derive(Debug)]
pub enum CudaSharedRoutedLayerState {
    Linear(CudaGatedDeltaState),
    Full(Box<CudaAffineGatedFullAttentionState>),
    Mamba2(CudaMamba2State),
    Attention(Box<DecodeAttentionBf16>),
}

#[derive(Clone, Debug)]
pub struct CudaSharedRoutedModelTemplate {
    backend: CudaBackend,
    decoder: DecoderConfig,
    embedding: CheckpointProjectionWeight,
    final_norm: CudaTensor,
    output: CheckpointProjectionWeight,
    layers: Vec<SharedRoutedLayerTemplate>,
    /// The multi-token prediction module, loaded for speculative decoding.
    prediction: Option<PredictionTemplate>,
    cache: CacheConfig,
    max_sequence_blocks: usize,
    /// Rows the decode batches' shared packed states and attention
    /// workspace are allocated for.
    decode_rows: usize,
    /// Distinguishes this model's shared decode resources from another
    /// model's in the runtime pools.
    identity: u64,
    norm_shift: f32,
    plans: Arc<Mutex<SharedRoutedExecutionPlanCache>>,
}

static NEXT_TEMPLATE_IDENTITY: AtomicU64 = AtomicU64::new(1);

impl CudaSharedRoutedModelTemplate {
    #[allow(clippy::too_many_arguments)]
    pub fn from_tensors(
        backend: &CudaBackend,
        decoder: &DecoderConfig,
        semantic: &SemanticModelSpec,
        tensors: &CudaTensorSet,
        catalog: &TensorCatalog,
        bindings: &WeightBindingPlan,
        cache: CacheConfig,
        max_sequence_blocks: usize,
        speculation: Option<Speculation>,
    ) -> Result<Self> {
        let hybrid = decoder.state_space.is_some();
        if hybrid {
            load::validate_state_space_composition(&semantic.decoder.layers, max_sequence_blocks)?;
        } else {
            load::validate_composition(semantic, max_sequence_blocks)?;
        }
        let boundary = bindings.decoder_boundary()?;
        let embedding = CheckpointProjectionWeight::load_binding_prepared(
            backend,
            tensors,
            boundary.embedding,
        )?;
        embedding.affine_format(1, decoder.hidden_size, decoder.vocab_size)?;
        let output = if decoder.tie_word_embeddings {
            embedding.clone()
        } else {
            CheckpointProjectionWeight::load_binding_prepared(backend, tensors, boundary.output)?
        };
        output.affine_format(1, decoder.hidden_size, decoder.vocab_size)?;
        let final_norm = required_norm(tensors, &boundary.final_norm.source, decoder.hidden_size)?;
        let (norm_shift, layers, prediction) = if hybrid {
            let load = load::HybridLoad {
                backend,
                decoder,
                tensors,
                catalog,
                cache,
                max_sequence_blocks,
            };
            let layers = semantic
                .decoder
                .layers
                .iter()
                .enumerate()
                .map(|(layer, spec)| {
                    load.layer(layer, spec, bindings.state_space_hybrid_layer(layer)?)
                })
                .collect::<Result<Vec<_>>>()?;
            // Speculation drafts only with a checkpoint's prediction module.
            let prediction = speculation
                .filter(|_| !semantic.decoder.prediction_layers.is_empty())
                .map(|_| PredictionTemplate::load(&load, semantic, bindings))
                .transpose()?;
            (0.0, layers, prediction)
        } else {
            let norm_shift = infer_norm_shift(tensors, decoder, bindings)?;
            let layers = (0..decoder.num_hidden_layers)
                .map(|layer| {
                    build_layer(
                        backend,
                        decoder,
                        tensors,
                        catalog,
                        layer,
                        bindings.mixed_decoder_layer(layer)?,
                        norm_shift,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            (norm_shift, layers, None)
        };
        Ok(Self {
            backend: backend.clone(),
            decoder: decoder.clone(),
            embedding,
            final_norm,
            output,
            layers,
            prediction,
            cache,
            max_sequence_blocks,
            decode_rows: 1,
            identity: NEXT_TEMPLATE_IDENTITY.fetch_add(1, Ordering::Relaxed),
            norm_shift,
            plans: Arc::new(Mutex::new(SharedRoutedExecutionPlanCache::new())),
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

    fn prepare_embedding(&self) -> Result<boundary::SharedRoutedEmbedding> {
        boundary::SharedRoutedEmbedding::new(
            &self.backend,
            self.decoder.hidden_size,
            self.decoder.vocab_size,
            &self.embedding,
        )
    }

    fn prepare_output_head(&self, tokens: usize) -> Result<boundary::SharedRoutedOutputHead> {
        boundary::SharedRoutedOutputHead::new(
            &self.backend,
            tokens,
            self.decoder.hidden_size,
            self.decoder.vocab_size,
            &self.output,
        )
    }

    #[must_use]
    pub const fn cache_config(&self) -> CacheConfig {
        self.cache
    }

    pub fn instantiate(&self) -> Result<CudaSharedRoutedModelSession> {
        let caches = self.allocate_shared_kv()?;
        self.instantiate_with_caches(&caches)
    }

    pub(crate) fn instantiate_with_caches(
        &self,
        caches: &[Option<PagedKvCache>],
    ) -> Result<CudaSharedRoutedModelSession> {
        CudaSharedRoutedModelSession::new(self, caches)
    }

    /// Whether the template loaded a prediction module to draft with.
    #[must_use]
    pub const fn drafts(&self) -> bool {
        self.prediction.is_some()
    }

    #[must_use]
    pub const fn final_norm_weight(&self) -> &CudaTensor {
        &self.final_norm
    }
}
