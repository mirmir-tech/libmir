mod execution;
mod scratch;
#[cfg(test)]
mod tests;
mod validation;
mod weights;

pub use execution::CudaAffineGatedFullAttentionExecution;
use models::layout::{AttentionLayerType, AttentionOutput, DecoderConfig, RotaryEmbeddingLayout};
use runtime::kv::KvStorageSpec;
pub use weights::AffineGatedFullAttentionWeights;

use super::{CudaBackend, PagedAttentionBf16, PagedKvCache};
use crate::{CudaTensorSet, Error, Result};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AffineGatedFullAttentionConfig {
    pub hidden_size: usize,
    pub query_heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_sections: [usize; 3],
    pub rope_interleaved: bool,
    pub rope_theta: f32,
    pub attention_scale: f32,
    pub rms_norm_epsilon: f32,
    pub norm_weight_shift: f32,
    pub group_size: usize,
    pub bits: usize,
}

impl AffineGatedFullAttentionConfig {
    pub fn from_decoder(
        decoder: &DecoderConfig,
        layer: usize,
        group_size: usize,
        bits: usize,
        norm_weight_shift: f32,
    ) -> Result<Self> {
        if decoder.layer_type(layer) != AttentionLayerType::Full {
            return Err(Error::UnsupportedDecoderLayer(format!(
                "layer {layer} is not full attention"
            )));
        }
        if decoder.attention_output != AttentionOutput::Gated {
            return Err(Error::UnsupportedDecoderLayer(
                "full attention output has no parsed gate".into(),
            ));
        }
        let head_dim = decoder.layer_head_dim(layer);
        let factor = decoder
            .partial_rotary_factor_for_layer(layer)
            .or(decoder.partial_rotary_factor)
            .unwrap_or(1.0);
        let head_dim_scalar = head_dim.to_string().parse::<f64>()?;
        let rotary_dim = (head_dim_scalar * factor)
            .round()
            .to_string()
            .parse()
            .map_err(|_| Error::InvalidDecoderKernel("invalid parsed rotary dimension"))?;
        let (rope_sections, rope_interleaved) = sections(&decoder.rope_layout, rotary_dim)?;
        let scale = decoder.attention_scale.unwrap_or_else(|| head_dim_scalar.sqrt().recip());
        let config = Self {
            hidden_size: decoder.hidden_size,
            query_heads: decoder.num_attention_heads,
            key_value_heads: decoder.layer_key_value_heads(layer),
            head_dim,
            rotary_dim,
            rope_sections,
            rope_interleaved,
            rope_theta: decoder
                .rope_theta_for_layer(layer)
                .unwrap_or(10_000.0)
                .to_string()
                .parse()?,
            attention_scale: scale.to_string().parse()?,
            rms_norm_epsilon: decoder.rms_norm_eps.to_string().parse()?,
            norm_weight_shift,
            group_size,
            bits,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn query_width(self) -> Result<usize> {
        checked(self.query_heads, self.head_dim)
    }

    pub fn key_value_width(self) -> Result<usize> {
        checked(self.key_value_heads, self.head_dim)
    }

    fn validate(self) -> Result<()> {
        let covered = self.rope_sections.iter().try_fold(0_usize, |total, value| {
            total
                .checked_add(*value)
                .ok_or(Error::InvalidDecoderKernel("MRoPE sections overflow"))
        })?;
        if self.hidden_size == 0
            || self.query_heads == 0
            || self.key_value_heads == 0
            || !self.query_heads.is_multiple_of(self.key_value_heads)
            || self.head_dim == 0
            || self.rotary_dim == 0
            || self.rotary_dim > self.head_dim
            || !self.rotary_dim.is_multiple_of(2)
            || covered != self.rotary_dim / 2
            || self.group_size == 0
            || !matches!(self.bits, 4 | 8)
            || !self.hidden_size.is_multiple_of(self.group_size)
            || !self.query_width()?.is_multiple_of(self.group_size)
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
            || !self.attention_scale.is_finite()
            || self.attention_scale <= 0.0
            || !self.rms_norm_epsilon.is_finite()
            || self.rms_norm_epsilon < 0.0
            || !self.norm_weight_shift.is_finite()
        {
            return Err(Error::InvalidDecoderKernel("invalid affine gated attention config"));
        }
        Ok(())
    }
}

fn sections(layout: &RotaryEmbeddingLayout, rotary_dim: usize) -> Result<([usize; 3], bool)> {
    let half = rotary_dim / 2;
    match layout {
        RotaryEmbeddingLayout::Standard => Ok(([half, 0, 0], false)),
        RotaryEmbeddingLayout::MultiSection(values)
        | RotaryEmbeddingLayout::InterleavedMultiSection(values) => {
            let values: [usize; 3] = values.clone().try_into().map_err(|_| {
                Error::UnsupportedDecoderLayer(
                    "MRoPE requires exactly three parsed sections".into(),
                )
            })?;
            Ok((values, matches!(layout, RotaryEmbeddingLayout::InterleavedMultiSection(_))))
        },
    }
}

#[derive(Clone, Debug)]
pub struct CudaAffineGatedFullAttention {
    backend: CudaBackend,
    config: AffineGatedFullAttentionConfig,
    weights: AffineGatedFullAttentionWeights,
}

impl CudaAffineGatedFullAttention {
    pub fn from_tensors(
        backend: &CudaBackend,
        tensors: &CudaTensorSet,
        prefix: &str,
        config: AffineGatedFullAttentionConfig,
    ) -> Result<Self> {
        Self::new(backend, config, AffineGatedFullAttentionWeights::load(tensors, prefix)?)
    }

    pub fn new(
        backend: &CudaBackend,
        config: AffineGatedFullAttentionConfig,
        weights: AffineGatedFullAttentionWeights,
    ) -> Result<Self> {
        config.validate()?;
        weights.validate(config)?;
        Ok(Self {
            backend: backend.clone(),
            config,
            weights,
        })
    }

    pub fn prepare(&self, tokens: usize) -> Result<CudaAffineGatedFullAttentionExecution> {
        CudaAffineGatedFullAttentionExecution::new(
            &self.backend, self.config, &self.weights, tokens,
        )
    }

    pub fn prepare_state(
        &self,
        layer: usize,
        storage: KvStorageSpec,
        max_sequence_blocks: usize,
    ) -> Result<CudaAffineGatedFullAttentionState> {
        if storage.kv_heads != self.config.key_value_heads
            || storage.key_head_dim != self.config.head_dim
            || storage.value_head_dim != self.config.head_dim
        {
            return Err(Error::InvalidPagedKv("gated attention KV geometry mismatch"));
        }
        let cache = self.backend.prepare_paged_kv(layer, storage)?;
        let attention = self.backend.prepare_paged_attention_bf16(
            &cache,
            self.config.query_heads,
            max_sequence_blocks,
        )?;
        Ok(CudaAffineGatedFullAttentionState { cache, attention })
    }
}

#[derive(Debug)]
pub struct CudaAffineGatedFullAttentionState {
    pub(super) cache: PagedKvCache,
    pub(super) attention: PagedAttentionBf16,
}

impl CudaAffineGatedFullAttentionState {
    #[must_use]
    pub fn storage_spec(&self) -> KvStorageSpec {
        self.cache.storage_spec()
    }
}

pub(super) fn checked(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or(Error::InvalidDecoderKernel("gated attention shape overflow"))
}
