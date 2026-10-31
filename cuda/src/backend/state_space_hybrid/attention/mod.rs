//! Attention layer without positional encoding: the paged decode attention
//! (its own `RMSNorm`, FP8 Q/K/V, attention, FP8 output projection) followed by
//! the residual and the optional feed-forward.

mod batch;
mod execution;

pub use execution::CudaAttentionHybridExecution;
use mircuda::{DeviceBuffer, bf16};
use models::{
    semantic::DecoderLayerSpec,
    weights::{AttentionProjectionBindings, TensorBinding},
};

use super::NormedFeedForward;
use crate::{
    AttentionLayerPolicy, CudaBackend, CudaTensor, CudaTensorSet, DecodeAttentionBf16,
    DecodeAttentionConfig, DecodeAttentionOutputWeight, DecodeAttentionWeights, DecodeQkvWeights,
    DirectFp8CheckpointWeight, Error, ExecutionPhase, PagedKvCache, ProjectionFormat, Result,
    backend::feed_forward::LayerNormConfig,
};

#[derive(Clone, Debug)]
pub struct CudaAttentionHybridLayer {
    backend: CudaBackend,
    norm: LayerNormConfig,
    config: DecodeAttentionConfig,
    weights: AttentionWeights,
    feed_forward: Option<NormedFeedForward>,
}

#[derive(Clone, Debug)]
struct AttentionWeights {
    input_norm: CudaTensor,
    query: DirectFp8CheckpointWeight,
    key: DirectFp8CheckpointWeight,
    value: DirectFp8CheckpointWeight,
    output: DirectFp8CheckpointWeight,
    /// Query/key norm operand of the shared QKV kernels; unused because the
    /// layer normalizes neither.
    unused_norm: DeviceBuffer<bf16>,
}

impl CudaAttentionHybridLayer {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend) fn load(
        backend: &CudaBackend,
        tensors: &CudaTensorSet,
        layer: usize,
        spec: &DecoderLayerSpec,
        input_norm: &TensorBinding,
        projections: AttentionProjectionBindings<'_>,
        feed_forward: Option<NormedFeedForward>,
        policy: AttentionLayerPolicy,
    ) -> Result<Self> {
        if policy.projection_format != ProjectionFormat::DirectFp8 {
            return Err(Error::UnsupportedDecoderLayer(
                "hybrid attention requires direct FP8 projections".into(),
            ));
        }
        let config = DecodeAttentionConfig::from_semantic(layer, spec, policy)?;
        let load = |binding| DirectFp8CheckpointWeight::load_binding(tensors, binding);
        let input_norm = tensors
            .get(&input_norm.source)
            .cloned()
            .ok_or_else(|| Error::MissingTensor(input_norm.source.clone()))?;
        let unused_norm = backend
            .inner
            .pool
            .allocate_zeroed(&backend.inner.stream, config.cache.key_head_dim)?;
        Ok(Self {
            backend: backend.clone(),
            norm: LayerNormConfig {
                hidden_size: policy.hidden_size,
                rms_norm_epsilon: config.rms_norm_epsilon,
                norm_weight_shift: 0.0,
            },
            config,
            weights: AttentionWeights {
                input_norm,
                query: load(projections.query)?,
                key: load(projections.key)?,
                value: load(projections.value)?,
                output: load(projections.output)?,
                unused_norm,
            },
            feed_forward,
        })
    }

    #[must_use]
    pub const fn config(&self) -> DecodeAttentionConfig {
        self.config
    }

    /// The layer's per-sequence attention over the model's shared pages,
    /// whose block count may differ from the one the layer was loaded with.
    pub fn prepare_state_with_cache(&self, cache: PagedKvCache) -> Result<DecodeAttentionBf16> {
        let mut config = self.config;
        config.cache.cache.block_count = cache.storage_spec().cache.block_count;
        DecodeAttentionBf16::new_with_cache_and_weights(
            &self.backend,
            config,
            cache,
            Some(self.weights.view()),
        )
    }

    /// K/V geometry of the layer over a cache of `cache` blocks.
    pub(in crate::backend) const fn storage(
        &self,
        cache: runtime::kv::CacheConfig,
    ) -> runtime::kv::KvStorageSpec {
        let mut storage = self.config.cache;
        storage.cache = cache;
        storage
    }

    pub fn prepare(&self, tokens: usize) -> Result<CudaAttentionHybridExecution> {
        let phase = if tokens == 1 {
            ExecutionPhase::Decode
        } else {
            ExecutionPhase::Prefill
        };
        self.prepare_phase(tokens, phase)
    }

    pub(crate) fn prepare_phase(
        &self,
        tokens: usize,
        phase: ExecutionPhase,
    ) -> Result<CudaAttentionHybridExecution> {
        CudaAttentionHybridExecution::new(self, tokens, phase)
    }
}

impl AttentionWeights {
    fn view(&self) -> DecodeAttentionWeights<'_> {
        DecodeAttentionWeights {
            input_norm: &self.input_norm,
            qkv: DecodeQkvWeights::DirectFp8([&self.query, &self.key, &self.value]),
            query_norm: &self.unused_norm,
            key_norm: &self.unused_norm,
            output: DecodeAttentionOutputWeight::DirectFp8(&self.output),
        }
    }
}
