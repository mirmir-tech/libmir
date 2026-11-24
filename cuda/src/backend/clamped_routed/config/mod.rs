//! The clamped-routed CUDA geometry, from the plan admission accepted.

use architecture::{cuda::CudaDecoderPlan, lowering::Activation};
use models::semantic::SemanticModelSpec;
use runtime::kv::{CacheConfig, KvStorageSpec};

use crate::{Error, Result};

#[derive(Clone, Copy, Debug)]
pub(super) struct ClampedRoutedConfig {
    pub vocab: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub experts: usize,
    pub top_k: usize,
    pub epsilon: f32,
    pub scale: f32,
    pub theta: f32,
    pub factor: f32,
    pub initial_context: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub rope_truncate: bool,
    pub swiglu_limit: f32,
}

impl ClampedRoutedConfig {
    /// The geometry of `spec` run as `plan`, the plan admission accepted for
    /// the clamped-routed runtime: every layer is the plan's repeated layer.
    pub(super) fn from_admitted(plan: &CudaDecoderPlan, spec: &SemanticModelSpec) -> Result<Self> {
        let layer = plan.repeated().ok_or_else(|| {
            Error::State("the clamped-routed runtime was admitted without a repeated layer".into())
        })?;
        let Activation::SwiGlu { clamp: Some(clamp) } = layer.experts.activation else {
            return Err(Error::State(
                "the clamped-routed runtime was admitted without clamped SwiGLU experts".into(),
            ));
        };
        let rotary = layer.rotary;
        Ok(Self {
            vocab: spec.decoder.vocab_size,
            hidden: spec.decoder.hidden_size,
            intermediate: layer.intermediate,
            query_heads: layer.query_heads,
            kv_heads: layer.key_value_heads,
            head_dim: layer.head_dim,
            experts: layer.experts.count,
            top_k: layer.experts.top_k,
            epsilon: narrow(layer.epsilon)?,
            scale: narrow(layer.scale)?,
            theta: narrow(rotary.theta)?,
            factor: narrow(rotary.factor)?,
            initial_context: rotary.original_context.to_string().parse()?,
            beta_fast: narrow(rotary.beta_fast)?,
            beta_slow: narrow(rotary.beta_slow)?,
            rope_truncate: rotary.truncate,
            swiglu_limit: narrow(clamp.limit)?,
        })
    }

    pub(super) fn storage(self, cache: CacheConfig) -> KvStorageSpec {
        KvStorageSpec::new(cache, self.kv_heads, self.head_dim)
    }
}

/// A model constant at the kernels' single precision.
fn narrow(value: f64) -> Result<f32> {
    Ok(value.to_string().parse()?)
}

#[cfg(test)]
mod tests;
