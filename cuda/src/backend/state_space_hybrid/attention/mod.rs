//! Attention layer without positional encoding: the paged decode attention
//! (its own `RMSNorm`, FP8 or BF16 Q/K/V, attention, output projection)
//! followed by the residual and the optional feed-forward.

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
    AttentionLayerPolicy, Bf16LinearPackWeights, CudaBackend, CudaTensor, CudaTensorSet,
    DecodeAttentionBf16, DecodeAttentionConfig, DecodeAttentionOutputWeight,
    DecodeAttentionWeights, DecodeQkvWeights, DirectFp8CheckpointWeight, Error, ExecutionPhase,
    PagedKvCache, ProjectionFormat, Result, backend::feed_forward::LayerNormConfig,
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
    projections: Projections,
    /// Query/key norm operand of the shared QKV kernels; unused because the
    /// layer normalizes neither.
    unused_norm: DeviceBuffer<bf16>,
}

/// Decoder layers carry FP8 projections; the multi-token prediction
/// module's attention is BF16.
#[derive(Clone, Debug)]
enum Projections {
    /// Q, K, V, then the output projection.
    DirectFp8(Box<[DirectFp8CheckpointWeight; 4]>),
    Bf16 {
        qkv: Bf16LinearPackWeights<3>,
        output: CudaTensor,
    },
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
        let config = DecodeAttentionConfig::from_semantic(layer, spec, policy)?;
        let projections =
            Projections::load(backend, tensors, policy.projection_format, projections)?;
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
            weights: AttentionWeights { input_norm, projections, unused_norm },
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
        let (qkv, output) = match &self.projections {
            Projections::DirectFp8(weights) => {
                let [query, key, value, output] = &**weights;
                (
                    DecodeQkvWeights::DirectFp8([query, key, value]),
                    DecodeAttentionOutputWeight::DirectFp8(output),
                )
            },
            Projections::Bf16 { qkv, output } => {
                (DecodeQkvWeights::Bf16(qkv), DecodeAttentionOutputWeight::Bf16(output))
            },
        };
        DecodeAttentionWeights {
            input_norm: &self.input_norm,
            qkv,
            query_norm: &self.unused_norm,
            key_norm: &self.unused_norm,
            output,
        }
    }
}

impl Projections {
    fn load(
        backend: &CudaBackend,
        tensors: &CudaTensorSet,
        format: ProjectionFormat,
        bindings: AttentionProjectionBindings<'_>,
    ) -> Result<Self> {
        match format {
            ProjectionFormat::DirectFp8 => {
                let load = |binding| DirectFp8CheckpointWeight::load_binding(tensors, binding);
                Ok(Self::DirectFp8(Box::new([
                    load(bindings.query)?,
                    load(bindings.key)?,
                    load(bindings.value)?,
                    load(bindings.output)?,
                ])))
            },
            ProjectionFormat::Bf16 => {
                let get = |binding: &TensorBinding| {
                    tensors
                        .get(&binding.source)
                        .ok_or_else(|| Error::MissingTensor(binding.source.clone()))
                };
                Ok(Self::Bf16 {
                    qkv: backend.pack_bf16_linears([
                        get(bindings.query)?,
                        get(bindings.key)?,
                        get(bindings.value)?,
                    ])?,
                    output: get(bindings.output)?.clone(),
                })
            },
            _ => Err(Error::UnsupportedDecoderLayer(
                "hybrid attention requires direct FP8 or BF16 projections".into(),
            )),
        }
    }
}
