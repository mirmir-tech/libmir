use super::AffineGatedFullAttentionConfig;
use crate::{AffineQuantizedWeight, CudaTensor, CudaTensorDType, CudaTensorSet, Error, Result};

#[derive(Clone, Debug)]
pub struct AffineGatedFullAttentionWeights {
    pub query: AffineQuantizedWeight,
    pub key: AffineQuantizedWeight,
    pub value: AffineQuantizedWeight,
    pub output: AffineQuantizedWeight,
    pub query_norm: CudaTensor,
    pub key_norm: CudaTensor,
}

impl AffineGatedFullAttentionWeights {
    pub fn load(tensors: &CudaTensorSet, prefix: &str) -> Result<Self> {
        Ok(Self {
            query: AffineQuantizedWeight::load(tensors, &format!("{prefix}.q_proj"))?,
            key: AffineQuantizedWeight::load(tensors, &format!("{prefix}.k_proj"))?,
            value: AffineQuantizedWeight::load(tensors, &format!("{prefix}.v_proj"))?,
            output: AffineQuantizedWeight::load(tensors, &format!("{prefix}.o_proj"))?,
            query_norm: required(tensors, &format!("{prefix}.q_norm.weight"))?,
            key_norm: required(tensors, &format!("{prefix}.k_norm.weight"))?,
        })
    }

    pub fn load_bindings(
        tensors: &CudaTensorSet,
        bindings: GatedSoftmaxBindings<'_>,
    ) -> Result<Self> {
        Ok(Self {
            query: AffineQuantizedWeight::load_binding(tensors, bindings.query)?,
            key: AffineQuantizedWeight::load_binding(tensors, bindings.key)?,
            value: AffineQuantizedWeight::load_binding(tensors, bindings.value)?,
            output: AffineQuantizedWeight::load_binding(tensors, bindings.output)?,
            query_norm: required(tensors, &bindings.query_norm.source)?,
            key_norm: required(tensors, &bindings.key_norm.source)?,
        })
    }

    pub(super) fn validate(&self, config: AffineGatedFullAttentionConfig) -> Result<()> {
        let query = config.query_width()?;
        let key_value = config.key_value_width()?;
        let projection = |weight: &AffineQuantizedWeight, input, output| {
            weight.validate(1, input, output, config.group_size, config.bits)
        };
        projection(&self.query, config.hidden_size, checked(query, 2)?)?;
        projection(&self.key, config.hidden_size, key_value)?;
        projection(&self.value, config.hidden_size, key_value)?;
        projection(&self.output, query, config.hidden_size)?;
        norm(&self.query_norm, config.head_dim)?;
        norm(&self.key_norm, config.head_dim)
    }
}

fn required(tensors: &CudaTensorSet, name: &str) -> Result<CudaTensor> {
    tensors.get(name).cloned().ok_or_else(|| Error::MissingTensor(name.into()))
}

fn norm(tensor: &CudaTensor, head_dim: usize) -> Result<()> {
    if tensor.shape() != [head_dim] {
        return Err(Error::InvalidQuantizedTensor {
            name: tensor.name().into(),
            expected: vec![head_dim],
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

fn checked(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or(Error::InvalidDecoderKernel("gated attention weight shape overflow"))
}
use models::weights::GatedSoftmaxBindings;
