mod batch;
mod boundary;
mod checkpoint;
mod load;
mod memory;
mod plan;
mod position;
mod session;
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
use runtime::kv::{CacheConfig, KvStorageSpec};

pub(in crate::backend) use self::batch::DecodeWorkspaceKey;
pub use self::{
    batch::{CudaSharedRoutedDecodeBatch, CudaSharedRoutedPrefillBatch},
    checkpoint::SharedRoutedCheckpoint,
    session::CudaSharedRoutedModelSession,
};
use self::{
    load::{build_layer, infer_norm_shift, required_norm},
    plan::SharedRoutedExecutionPlanCache,
};
use crate::{
    CudaAffineGatedDeltaMoeLayer, CudaAffineGatedFullAttentionMoeLayer,
    CudaAffineGatedFullAttentionState, CudaBackend, CudaGatedDeltaState, CudaTensor, CudaTensorSet,
    Error, PagedKvCache, Result, backend::linear::CheckpointProjectionWeight,
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

#[derive(Clone, Debug)]
pub struct CudaSharedRoutedModelTemplate {
    backend: CudaBackend,
    decoder: DecoderConfig,
    embedding: CheckpointProjectionWeight,
    final_norm: CudaTensor,
    output: CheckpointProjectionWeight,
    layers: Vec<SharedRoutedLayerTemplate>,
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
    ) -> Result<Self> {
        load::validate_composition(semantic, max_sequence_blocks)?;
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
        Ok(Self {
            backend: backend.clone(),
            decoder: decoder.clone(),
            embedding,
            final_norm,
            output,
            layers,
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

    fn cache_spec(&self) -> Result<KvStorageSpec> {
        self.layers
            .iter()
            .enumerate()
            .find_map(|(index, layer)| {
                matches!(layer, SharedRoutedLayerTemplate::Full(_)).then(|| {
                    KvStorageSpec::new(
                        self.cache,
                        self.decoder.layer_key_value_heads(index),
                        self.decoder.layer_head_dim(index),
                    )
                })
            })
            .ok_or(Error::InvalidPagedKv("shared-routed model has no full-attention cache"))
    }

    pub(crate) fn allocate_shared_kv(&self) -> Result<Vec<Option<PagedKvCache>>> {
        self.layers
            .iter()
            .enumerate()
            .map(|(index, layer)| match layer {
                SharedRoutedLayerTemplate::Linear(_) => Ok(None),
                SharedRoutedLayerTemplate::Full(_) => {
                    let storage = KvStorageSpec::new(
                        self.cache,
                        self.decoder.layer_key_value_heads(index),
                        self.decoder.layer_head_dim(index),
                    );
                    self.backend.prepare_paged_kv(index, storage).map(Some)
                },
            })
            .collect()
    }

    fn prepare_states(
        &self,
        caches: &[Option<PagedKvCache>],
    ) -> Result<Vec<CudaSharedRoutedLayerState>> {
        if caches.len() != self.layers.len() {
            return Err(Error::InvalidPagedKv("shared-routed cache count differs from layers"));
        }
        self.layers
            .iter()
            .zip(caches)
            .map(|(layer, cache)| match layer {
                SharedRoutedLayerTemplate::Linear(layer) => {
                    layer.prepare_state().map(CudaSharedRoutedLayerState::Linear)
                },
                SharedRoutedLayerTemplate::Full(layer) => layer
                    .prepare_state_with_cache(
                        cache.clone().ok_or(Error::InvalidPagedKv(
                            "shared-routed full-attention cache is missing",
                        ))?,
                        self.max_sequence_blocks,
                    )
                    .map(Box::new)
                    .map(CudaSharedRoutedLayerState::Full),
            })
            .collect()
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

    #[must_use]
    pub const fn final_norm_weight(&self) -> &CudaTensor {
        &self.final_norm
    }
}
