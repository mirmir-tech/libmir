use super::AffineGatedDeltaLayerConfig;
use crate::{AffineQuantizedWeight, CudaTensor, CudaTensorDType, CudaTensorSet, Error, Result};

#[derive(Clone, Debug)]
pub struct AffineGatedDeltaLayerWeights {
    pub qkv: AffineQuantizedWeight,
    pub gate: AffineQuantizedWeight,
    pub alpha: AffineQuantizedWeight,
    pub beta: AffineQuantizedWeight,
    pub output: AffineQuantizedWeight,
    pub convolution: CudaTensor,
    pub norm: CudaTensor,
    pub a_log: CudaTensor,
    pub dt_bias: CudaTensor,
}

impl AffineGatedDeltaLayerWeights {
    pub fn load(tensors: &CudaTensorSet, prefix: &str) -> Result<Self> {
        Ok(Self {
            qkv: AffineQuantizedWeight::load(tensors, &format!("{prefix}.in_proj_qkv"))?,
            gate: AffineQuantizedWeight::load(tensors, &format!("{prefix}.in_proj_z"))?,
            alpha: AffineQuantizedWeight::load(tensors, &format!("{prefix}.in_proj_a"))?,
            beta: AffineQuantizedWeight::load(tensors, &format!("{prefix}.in_proj_b"))?,
            output: AffineQuantizedWeight::load(tensors, &format!("{prefix}.out_proj"))?,
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
            qkv: AffineQuantizedWeight::load_binding(tensors, bindings.qkv)?,
            gate: AffineQuantizedWeight::load_binding(tensors, bindings.gate)?,
            alpha: AffineQuantizedWeight::load_binding(tensors, bindings.alpha)?,
            beta: AffineQuantizedWeight::load_binding(tensors, bindings.beta)?,
            output: AffineQuantizedWeight::load_binding(tensors, bindings.output)?,
            convolution: required(tensors, &bindings.convolution.source)?,
            norm: required(tensors, &bindings.norm.source)?,
            a_log: required(tensors, &bindings.decay_log.source)?,
            dt_bias: required(tensors, &bindings.time_bias.source)?,
        })
    }

    pub(super) fn validate(&self, config: AffineGatedDeltaLayerConfig) -> Result<()> {
        let mixed = config.mixed_width()?;
        let value = config.value_width()?;
        let projection = |weights: &AffineQuantizedWeight, input, output| {
            weights.validate(1, input, output, config.group_size, config.bits)
        };
        projection(&self.qkv, config.hidden_size, mixed)?;
        projection(&self.gate, config.hidden_size, value)?;
        projection(&self.alpha, config.hidden_size, config.value_heads)?;
        projection(&self.beta, config.hidden_size, config.value_heads)?;
        projection(&self.output, value, config.hidden_size)?;
        shape(&self.convolution, &[mixed, config.convolution_kernel_size, 1])?;
        shape(&self.norm, &[config.value_dim])?;
        shape(&self.a_log, &[config.value_heads])?;
        shape(&self.dt_bias, &[config.value_heads])?;
        for tensor in [&self.convolution, &self.norm, &self.a_log, &self.dt_bias] {
            dtype(tensor, CudaTensorDType::Bf16, "BF16")?;
        }
        Ok(())
    }
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
