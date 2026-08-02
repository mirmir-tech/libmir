use super::AffineGatedDeltaLayerConfig;
use crate::{
    AffineQuantizedWeight, CudaTensor, CudaTensorDType, CudaTensorSet, Error, Result,
    backend::linear::CheckpointProjectionWeight,
};

#[derive(Clone, Debug)]
pub struct AffineGatedDeltaLayerWeights {
    pub(in crate::backend) qkv: CheckpointProjectionWeight,
    pub(in crate::backend) gate: CheckpointProjectionWeight,
    pub(in crate::backend) alpha: CheckpointProjectionWeight,
    pub(in crate::backend) beta: CheckpointProjectionWeight,
    pub(in crate::backend) output: CheckpointProjectionWeight,
    pub convolution: CudaTensor,
    pub norm: CudaTensor,
    pub a_log: CudaTensor,
    pub dt_bias: CudaTensor,
}

impl AffineGatedDeltaLayerWeights {
    pub fn load(tensors: &CudaTensorSet, prefix: &str) -> Result<Self> {
        Ok(Self {
            qkv: affine(tensors, &format!("{prefix}.in_proj_qkv"))?,
            gate: affine(tensors, &format!("{prefix}.in_proj_z"))?,
            alpha: affine(tensors, &format!("{prefix}.in_proj_a"))?,
            beta: affine(tensors, &format!("{prefix}.in_proj_b"))?,
            output: affine(tensors, &format!("{prefix}.out_proj"))?,
            convolution: required(tensors, &format!("{prefix}.conv1d.weight"))?,
            norm: required(tensors, &format!("{prefix}.norm.weight"))?,
            a_log: required(tensors, &format!("{prefix}.A_log"))?,
            dt_bias: required(tensors, &format!("{prefix}.dt_bias"))?,
        })
    }

    pub fn load_bindings(
        tensors: &CudaTensorSet,
        bindings: LinearAttentionBindings<'_>,
    ) -> Result<Self> {
        Ok(Self {
            qkv: CheckpointProjectionWeight::load_binding(tensors, bindings.qkv)?,
            gate: CheckpointProjectionWeight::load_binding(tensors, bindings.gate)?,
            alpha: CheckpointProjectionWeight::load_binding(tensors, bindings.alpha)?,
            beta: CheckpointProjectionWeight::load_binding(tensors, bindings.beta)?,
            output: CheckpointProjectionWeight::load_binding(tensors, bindings.output)?,
            convolution: required(tensors, &bindings.convolution.source)?,
            norm: required(tensors, &bindings.norm.source)?,
            a_log: required(tensors, &bindings.decay_log.source)?,
            dt_bias: required(tensors, &bindings.time_bias.source)?,
        })
    }

    pub(super) fn validate(&self, config: AffineGatedDeltaLayerConfig) -> Result<()> {
        let mixed = config.mixed_width()?;
        let value = config.value_width()?;
        let projection = |weights: &CheckpointProjectionWeight, input, output| {
            weights.validate(1, input, output, config.group_size, config.bits)
        };
        projection(&self.qkv, config.hidden_size, mixed)?;
        projection(&self.gate, config.hidden_size, value)?;
        projection(&self.alpha, config.hidden_size, config.value_heads)?;
        projection(&self.beta, config.hidden_size, config.value_heads)?;
        projection(&self.output, value, config.hidden_size)?;
        convolution_shape(&self.convolution, mixed, config.convolution_kernel_size)?;
        shape(&self.norm, &[config.value_dim])?;
        shape(&self.a_log, &[config.value_heads])?;
        shape(&self.dt_bias, &[config.value_heads])?;
        for tensor in [&self.convolution, &self.norm, &self.a_log, &self.dt_bias] {
            dtype(tensor, CudaTensorDType::Bf16, "BF16")?;
        }
        Ok(())
    }
}

fn affine(tensors: &CudaTensorSet, name: &str) -> Result<CheckpointProjectionWeight> {
    AffineQuantizedWeight::load(tensors, name).map(CheckpointProjectionWeight::Affine)
}

fn convolution_shape(tensor: &CudaTensor, channels: usize, kernel: usize) -> Result<()> {
    if matches!(tensor.shape(), [c, k, 1] if *c == channels && *k == kernel)
        || matches!(tensor.shape(), [c, 1, k] if *c == channels && *k == kernel)
    {
        return Ok(());
    }
    Err(Error::InvalidQuantizedTensor {
        name: tensor.name().into(),
        expected: vec![channels, kernel, 1],
        actual: tensor.shape().to_vec(),
    })
}

fn required(tensors: &CudaTensorSet, name: &str) -> Result<CudaTensor> {
    tensors.get(name).cloned().ok_or_else(|| Error::MissingTensor(name.into()))
}

fn shape(tensor: &CudaTensor, expected: &[usize]) -> Result<()> {
    if tensor.shape() != expected {
        return Err(Error::InvalidQuantizedTensor {
            name: tensor.name().into(),
            expected: expected.to_vec(),
            actual: tensor.shape().to_vec(),
        });
    }
    Ok(())
}

fn dtype(
    tensor: &CudaTensor,
    expected: CudaTensorDType,
    expected_name: &'static str,
) -> Result<()> {
    if tensor.dtype() != expected {
        return Err(Error::DTypeMismatch {
            name: tensor.name().into(),
            expected: expected_name,
        });
    }
    Ok(())
}
use models::weights::LinearAttentionBindings;
