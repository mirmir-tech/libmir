//! Multi-head latent attention decoder layer (`DeepSeek` V2/V3, `Moonlight`):
//! input `RMSNorm`, query and key/value latent projections, absorbed
//! attention over the layer's latent cache, output projection, then the
//! residual and the feed-forward.

mod execution;
mod feed_forward;
mod rotary;
mod weights;

pub use execution::CudaLatentAttentionExecution;
use mircuda::DeviceBuffer;
use models::{
    layout::{DecoderConfig, RotaryPairing},
    semantic::{DecoderLayerSpec, MixerSpec, RotaryPairingSpec},
    weights::{LatentDecoderLayerBindings, TensorCatalog},
};
use runtime::kv::{CacheConfig, KvStorageSpec};

use self::weights::LatentWeights;
use super::state_space_hybrid::NormedFeedForward;
use crate::{
    CudaBackend, CudaTensorSet, Error, ExecutionPhase, PagedKvCache, Result,
    backend::feed_forward::LayerNormConfig, kernels::LatentAttentionSpec,
};

/// Geometry and constants shared by every execution of a layer.
#[derive(Clone, Copy, Debug)]
pub(super) struct LatentGeometry {
    pub(super) heads: usize,
    pub(super) query_rank: Option<usize>,
    pub(super) latent_rank: usize,
    pub(super) rope_dim: usize,
    pub(super) nope_dim: usize,
    pub(super) value_dim: usize,
    pub(super) rotary_pairing: RotaryPairing,
    pub(super) latent_epsilon: f32,
    pub(super) rope_scale: f32,
    pub(super) softmax_scale: f32,
}

impl LatentGeometry {
    pub(super) const fn row_width(&self) -> usize {
        self.latent_rank + self.rope_dim
    }

    pub(super) const fn query_width(&self) -> usize {
        self.heads * (self.nope_dim + self.rope_dim)
    }

    /// Kernel geometry for `tokens` query tokens over `block_size` pages.
    pub(super) fn kernel_spec(&self, tokens: usize, block_size: usize) -> LatentAttentionSpec {
        // Decode splits every query's tokens until about a hundred blocks run
        // at once; prefill has enough query tokens without splitting.
        let blocks = self.heads.div_ceil(8) * tokens;
        LatentAttentionSpec {
            heads: self.heads,
            latent_rank: self.latent_rank,
            rope_dim: self.rope_dim,
            nope_dim: self.nope_dim,
            value_dim: self.value_dim,
            rotary_pairing: self.rotary_pairing,
            block_size,
            max_queries: tokens,
            splits: (96 / blocks).clamp(1, 32),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CudaLatentAttentionLayer {
    backend: CudaBackend,
    norm: LayerNormConfig,
    geometry: LatentGeometry,
    weights: LatentWeights,
    /// Tokens of one cache block.
    block_size: usize,
    /// `[rope_dim / 2]` rotary inverse frequencies.
    inverse_frequencies: DeviceBuffer<f32>,
    feed_forward: Option<NormedFeedForward>,
}

/// One session's view of the layer: the model's shared latent pages.
#[derive(Debug)]
pub struct LatentAttentionState {
    pub(crate) cache: PagedKvCache,
}

impl CudaLatentAttentionLayer {
    pub(in crate::backend) fn load(
        backend: &CudaBackend,
        decoder: &DecoderConfig,
        tensors: &CudaTensorSet,
        catalog: &TensorCatalog,
        spec: &DecoderLayerSpec,
        bindings: LatentDecoderLayerBindings<'_>,
        cache: CacheConfig,
    ) -> Result<Self> {
        let MixerSpec::LatentAttention(latent) = &spec.mixer else {
            return Err(Error::UnsupportedDecoderLayer(format!(
                "layer {} is not a latent attention layer",
                spec.index
            )));
        };
        let (frequencies, rope_scale) =
            rotary::frequencies(&latent.position, latent.rope_head_dim)?;
        let epsilon: f32 = decoder.rms_norm_eps.to_string().parse()?;
        #[allow(clippy::cast_possible_truncation)]
        let geometry = LatentGeometry {
            heads: latent.query_heads,
            query_rank: latent.query_rank,
            latent_rank: latent.key_value_rank,
            rope_dim: latent.rope_head_dim,
            nope_dim: latent.nope_head_dim,
            value_dim: latent.value_head_dim,
            rotary_pairing: match latent.rotary_pairing {
                RotaryPairingSpec::Interleaved => RotaryPairing::Interleaved,
                RotaryPairingSpec::Halves => RotaryPairing::Halves,
            },
            latent_epsilon: latent.latent_normalization.epsilon as f32,
            rope_scale,
            softmax_scale: latent.scale as f32,
        };
        let feed_forward = NormedFeedForward {
            norm: tensors
                .get(&bindings.post_attention_norm.source)
                .cloned()
                .ok_or_else(|| Error::MissingTensor(bindings.post_attention_norm.source.clone()))?,
            block: feed_forward::load(
                backend,
                decoder,
                tensors,
                catalog,
                &spec.feed_forward,
                bindings.feed_forward,
            )?,
        };
        let mut staging = backend.inner.context.allocate_pinned(frequencies.len())?;
        staging.copy_from_slice(&frequencies)?;
        let mut inverse_frequencies =
            backend.inner.pool.allocate(&backend.inner.stream, frequencies.len())?;
        backend.inner.stream.copy_to_device(&mut staging, &mut inverse_frequencies)?;
        Ok(Self {
            backend: backend.clone(),
            norm: LayerNormConfig {
                hidden_size: decoder.hidden_size,
                rms_norm_epsilon: epsilon,
                norm_weight_shift: 0.0,
            },
            geometry,
            weights: LatentWeights::load(
                backend,
                tensors,
                bindings.input_norm,
                bindings.attention,
            )?,
            block_size: cache.block_size,
            inverse_frequencies,
            feed_forward: Some(feed_forward),
        })
    }

    /// The layer's latent cache over `cache` blocks: one BF16 row of latent
    /// and rotary key per token.
    pub(in crate::backend) fn storage(&self, cache: CacheConfig) -> KvStorageSpec {
        KvStorageSpec {
            value_head_dim: 0,
            ..KvStorageSpec::new(cache, 1, self.geometry.row_width())
        }
    }

    pub fn prepare_state_with_cache(&self, cache: PagedKvCache) -> Result<LatentAttentionState> {
        if !cache.is_latent() {
            return Err(Error::InvalidPagedKv("latent attention needs a latent cache"));
        }
        Ok(LatentAttentionState { cache })
    }

    pub(crate) fn prepare_phase(
        &self,
        tokens: usize,
        phase: ExecutionPhase,
    ) -> Result<CudaLatentAttentionExecution> {
        CudaLatentAttentionExecution::new(self, tokens, phase)
    }
}
