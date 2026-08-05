mod batch;
mod execution;
mod scratch;

pub use execution::CudaAffineGatedFullAttentionMoeExecution;
use runtime::kv::KvStorageSpec;

use crate::{
    AffineGatedFullAttentionConfig, AffineSharedExpertMoeConfig, CudaAffineGatedFullAttention,
    CudaAffineGatedFullAttentionState, CudaAffineSharedExpertMoe, CudaBackend, CudaTensor,
    CudaTensorDType, CudaTensorSet, Error, ExecutionPhase, Result,
    kernels::BatchedSplitAttentionWorkspace,
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
        Self::new(
            backend,
            config,
            CudaAffineGatedFullAttention::from_tensors(
                backend,
                tensors,
                &format!("{prefix}.self_attn"),
                config.attention,
            )?,
            CudaAffineSharedExpertMoe::from_tensors(
                backend,
                tensors,
                &format!("{prefix}.mlp"),
                config.moe,
            )?,
            norm(tensors, &format!("{prefix}.input_layernorm.weight"), config)?,
            norm(tensors, &format!("{prefix}.post_attention_layernorm.weight"), config)?,
        )
    }

    pub fn new(
        backend: &CudaBackend,
        config: AffineGatedFullAttentionMoeLayerConfig,
        attention: CudaAffineGatedFullAttention,
        moe: CudaAffineSharedExpertMoe,
        input_norm: CudaTensor,
        post_attention_norm: CudaTensor,
    ) -> Result<Self> {
        config.validate()?;
        validate_norm(&input_norm, config)?;
        validate_norm(&post_attention_norm, config)?;
        Ok(Self {
            backend: backend.clone(),
            config,
            attention,
            moe,
            input_norm,
            post_attention_norm,
        })
    }

    pub fn prepare(&self, tokens: usize) -> Result<CudaAffineGatedFullAttentionMoeExecution> {
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
    ) -> Result<CudaAffineGatedFullAttentionMoeExecution> {
        self.prepare_phase_with_workspace(tokens, phase, None)
    }

    pub(crate) fn prepare_phase_with_workspace(
        &self,
        tokens: usize,
        phase: ExecutionPhase,
        workspace: Option<BatchedSplitAttentionWorkspace>,
    ) -> Result<CudaAffineGatedFullAttentionMoeExecution> {
        let mut execution = CudaAffineGatedFullAttentionMoeExecution::new(
            &self.backend,
            self.config,
            &self.attention,
            &self.moe,
            &self.input_norm,
            &self.post_attention_norm,
            tokens,
            phase,
        )?;
        execution.attention.batch_workspace = workspace;
        Ok(execution)
    }

    pub fn prepare_state(
        &self,
        layer: usize,
        storage: KvStorageSpec,
        max_sequence_blocks: usize,
    ) -> Result<CudaAffineGatedFullAttentionState> {
        self.attention.prepare_state(layer, storage, max_sequence_blocks)
    }

    pub(crate) fn prepare_state_with_cache(
        &self,
        cache: crate::PagedKvCache,
        max_sequence_blocks: usize,
    ) -> Result<CudaAffineGatedFullAttentionState> {
        self.attention.prepare_state_with_cache(cache, max_sequence_blocks)
    }
}

fn norm(
    tensors: &CudaTensorSet,
    name: &str,
    config: AffineGatedFullAttentionMoeLayerConfig,
) -> Result<CudaTensor> {
    let tensor = tensors.get(name).cloned().ok_or_else(|| Error::MissingTensor(name.into()))?;
    validate_norm(&tensor, config)?;
    Ok(tensor)
}

fn validate_norm(
    tensor: &CudaTensor,
    config: AffineGatedFullAttentionMoeLayerConfig,
) -> Result<()> {
    if tensor.shape() != [config.attention.hidden_size] {
        return Err(Error::InvalidQuantizedTensor {
            name: tensor.name().into(),
            expected: vec![config.attention.hidden_size],
            actual: tensor.shape().to_vec(),
        });
    }
    if tensor.dtype() != CudaTensorDType::Bf16 {
        return Err(Error::DTypeMismatch {
            name: tensor.name().into(),
            expected: "BF16",
        });
    }
    Ok(())
}
