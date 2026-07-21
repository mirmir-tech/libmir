mod execution;
mod scratch;
#[cfg(all(test, target_os = "linux"))]
mod tests;

pub use execution::CudaAffineGatedDeltaMoeExecution;

use crate::{
    AffineGatedDeltaLayerConfig, AffineSharedExpertMoeConfig, CudaAffineGatedDeltaLayer,
    CudaAffineSharedExpertMoe, CudaBackend, CudaGatedDeltaState, CudaTensor, CudaTensorDType,
    CudaTensorSet, Error, Result,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AffineGatedDeltaMoeLayerConfig {
    pub attention: AffineGatedDeltaLayerConfig,
    pub moe: AffineSharedExpertMoeConfig,
    pub rms_norm_epsilon: f32,
    pub norm_weight_shift: f32,
}

impl AffineGatedDeltaMoeLayerConfig {
    fn validate(self) -> Result<()> {
        if self.attention.hidden_size != self.moe.hidden_size
            || !self.rms_norm_epsilon.is_finite()
            || self.rms_norm_epsilon < 0.0
            || !self.norm_weight_shift.is_finite()
        {
            return Err(Error::InvalidDecoderKernel("invalid affine hybrid layer config"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct CudaAffineGatedDeltaMoeLayer {
    backend: CudaBackend,
    config: AffineGatedDeltaMoeLayerConfig,
    attention: CudaAffineGatedDeltaLayer,
    moe: CudaAffineSharedExpertMoe,
    input_norm: CudaTensor,
    post_attention_norm: CudaTensor,
}

impl CudaAffineGatedDeltaMoeLayer {
    pub fn from_tensors(
        backend: &CudaBackend,
        tensors: &CudaTensorSet,
        prefix: &str,
        config: AffineGatedDeltaMoeLayerConfig,
    ) -> Result<Self> {
        Self::new(
            backend,
            config,
            CudaAffineGatedDeltaLayer::from_tensors(
                backend,
                tensors,
                &format!("{prefix}.linear_attn"),
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
        config: AffineGatedDeltaMoeLayerConfig,
        attention: CudaAffineGatedDeltaLayer,
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

    pub fn prepare(&self, tokens: usize) -> Result<CudaAffineGatedDeltaMoeExecution> {
        CudaAffineGatedDeltaMoeExecution::new(
            &self.backend,
            self.config,
            &self.attention,
            &self.moe,
            &self.input_norm,
            &self.post_attention_norm,
            tokens,
        )
    }

    pub fn prepare_state(&self) -> Result<CudaGatedDeltaState> {
        self.attention.prepare_state()
    }
}

fn norm(
    tensors: &CudaTensorSet,
    name: &str,
    config: AffineGatedDeltaMoeLayerConfig,
) -> Result<CudaTensor> {
    let tensor = tensors.get(name).cloned().ok_or_else(|| Error::MissingTensor(name.into()))?;
    validate_norm(&tensor, config)?;
    Ok(tensor)
}

fn validate_norm(tensor: &CudaTensor, config: AffineGatedDeltaMoeLayerConfig) -> Result<()> {
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
