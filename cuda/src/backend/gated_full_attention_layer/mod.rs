mod execution;
mod scratch;

pub use execution::CudaAffineGatedFullAttentionMoeExecution;
use runtime::kv::KvStorageSpec;

use crate::{
    AffineGatedFullAttentionConfig, AffineSharedExpertMoeConfig, CudaAffineGatedFullAttention,
    CudaAffineGatedFullAttentionState, CudaAffineSharedExpertMoe, CudaBackend, CudaTensor,
    CudaTensorDType, CudaTensorSet, Error, Result,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AffineGatedFullAttentionMoeLayerConfig {
    pub attention: AffineGatedFullAttentionConfig,
    pub moe: AffineSharedExpertMoeConfig,
    pub rms_norm_epsilon: f32,
    pub norm_weight_shift: f32,
}

impl AffineGatedFullAttentionMoeLayerConfig {
    fn validate(self) -> Result<()> {
        if self.attention.hidden_size != self.moe.hidden_size
            || !self.rms_norm_epsilon.is_finite()
            || self.rms_norm_epsilon < 0.0
            || !self.norm_weight_shift.is_finite()
        {
            return Err(Error::InvalidDecoderKernel(
                "invalid affine gated full-attention MoE layer config",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct CudaAffineGatedFullAttentionMoeLayer {
    backend: CudaBackend,
    config: AffineGatedFullAttentionMoeLayerConfig,
    attention: CudaAffineGatedFullAttention,
    moe: CudaAffineSharedExpertMoe,
    input_norm: CudaTensor,
    post_attention_norm: CudaTensor,
}

impl CudaAffineGatedFullAttentionMoeLayer {
    pub fn from_tensors(
        backend: &CudaBackend,
        tensors: &CudaTensorSet,
        prefix: &str,
        config: AffineGatedFullAttentionMoeLayerConfig,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            backend: backend.clone(),
            config,
            attention: CudaAffineGatedFullAttention::from_tensors(
                backend,
                tensors,
                &format!("{prefix}.self_attn"),
                config.attention,
            )?,
            moe: CudaAffineSharedExpertMoe::from_tensors(
                backend,
                tensors,
                &format!("{prefix}.mlp"),
                config.moe,
            )?,
            input_norm: norm(tensors, &format!("{prefix}.input_layernorm.weight"), config)?,
            post_attention_norm: norm(
                tensors,
                &format!("{prefix}.post_attention_layernorm.weight"),
                config,
            )?,
        })
    }

    pub fn prepare(&self, tokens: usize) -> Result<CudaAffineGatedFullAttentionMoeExecution> {
        CudaAffineGatedFullAttentionMoeExecution::new(
            &self.backend,
            self.config,
            &self.attention,
            &self.moe,
            &self.input_norm,
            &self.post_attention_norm,
            tokens,
        )
    }

    pub fn prepare_state(
        &self,
        layer: usize,
        storage: KvStorageSpec,
        max_sequence_blocks: usize,
    ) -> Result<CudaAffineGatedFullAttentionState> {
        self.attention.prepare_state(layer, storage, max_sequence_blocks)
    }
}

fn norm(
    tensors: &CudaTensorSet,
    name: &str,
    config: AffineGatedFullAttentionMoeLayerConfig,
) -> Result<CudaTensor> {
    let tensor = tensors.get(name).cloned().ok_or_else(|| Error::MissingTensor(name.into()))?;
    if tensor.shape() != [config.attention.hidden_size] {
        return Err(Error::InvalidQuantizedTensor {
            name: name.into(),
            expected: vec![config.attention.hidden_size],
            actual: tensor.shape().to_vec(),
        });
    }
    if tensor.dtype() != CudaTensorDType::Bf16 {
        return Err(Error::DTypeMismatch { name: name.into(), expected: "BF16" });
    }
    Ok(tensor)
}
