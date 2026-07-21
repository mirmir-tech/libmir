use models::weights::{TensorBinding, TensorStorage};

use crate::{
    AffineQuantizedConfig, AffineQuantizedTensors, CudaTensor, CudaTensorDType, CudaTensorSet,
    Error, Result,
};

#[derive(Clone, Debug)]
pub struct AffineQuantizedWeight {
    pub weight: CudaTensor,
    pub scales: CudaTensor,
    pub biases: CudaTensor,
}

impl AffineQuantizedWeight {
    pub fn load(tensors: &CudaTensorSet, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: required(tensors, &format!("{prefix}.weight"))?,
            scales: required(tensors, &format!("{prefix}.scales"))?,
            biases: required(tensors, &format!("{prefix}.biases"))?,
        })
    }

    pub(crate) fn load_binding(tensors: &CudaTensorSet, binding: &TensorBinding) -> Result<Self> {
        let TensorStorage::AffineQuantized { scales, biases: Some(biases), .. } = &binding.storage
        else {
            return Err(Error::InvalidQuantizedGemv(
                "binding is not an affine weight with zero-point biases",
            ));
        };
        Ok(Self {
            weight: required(tensors, &binding.source)?,
            scales: required(tensors, scales)?,
            biases: required(tensors, biases)?,
        })
    }

    pub(crate) fn validate(
        &self,
        matrices: usize,
        input: usize,
        output: usize,
        group_size: usize,
        bits: usize,
    ) -> Result<()> {
        let values_per_word = 32_usize
            .checked_div(bits)
            .filter(|value| *value != 0)
            .ok_or(Error::InvalidQuantizedGemv("invalid affine precision"))?;
        let packed = input / values_per_word;
        let groups = input / group_size;
        shape(&self.weight, expected(matrices, output, packed))?;
        shape(&self.scales, expected(matrices, output, groups))?;
        shape(&self.biases, expected(matrices, output, groups))?;
        dtype(&self.weight, CudaTensorDType::U32, "U32")?;
        dtype(&self.scales, CudaTensorDType::Bf16, "BF16")?;
        dtype(&self.biases, CudaTensorDType::Bf16, "BF16")
    }

    /// Derives affine precision and group geometry from tensor shapes.
    pub fn infer_config(
        &self,
        matrices: usize,
        input: usize,
        output: usize,
    ) -> Result<AffineQuantizedConfig> {
        let packed = trailing(&self.weight, matrices, output)?;
        let groups = trailing(&self.scales, matrices, output)?;
        if trailing(&self.biases, matrices, output)? != groups || groups == 0 || input == 0 {
            return Err(Error::InvalidQuantizedGemv("inconsistent affine group tensors"));
        }
        let packed_bits = packed
            .checked_mul(32)
            .ok_or(Error::InvalidQuantizedGemv("affine packed width overflow"))?;
        if !packed_bits.is_multiple_of(input) || !input.is_multiple_of(groups) {
            return Err(Error::InvalidQuantizedGemv("affine tensor shape is not integral"));
        }
        let config = AffineQuantizedConfig::new(input, output, input / groups, packed_bits / input);
        self.validate(matrices, input, output, config.group_size, config.bits)?;
        Ok(config)
    }

    pub(crate) const fn tensors(&self) -> AffineQuantizedTensors<'_> {
        AffineQuantizedTensors {
            weight: &self.weight,
            scales: &self.scales,
            biases: &self.biases,
        }
    }
}

fn trailing(tensor: &CudaTensor, matrices: usize, output: usize) -> Result<usize> {
    let shape = tensor.shape();
    let value = match (matrices, shape) {
        (1, [rows, trailing]) if *rows == output => Some(*trailing),
        (_, [banks, rows, trailing]) if *banks == matrices && *rows == output => Some(*trailing),
        _ => None,
    };
    value.ok_or_else(|| Error::InvalidQuantizedTensor {
        name: tensor.name().into(),
        expected: expected(matrices, output, 0),
        actual: shape.to_vec(),
    })
}

fn expected(matrices: usize, output: usize, trailing: usize) -> Vec<usize> {
    if matrices == 1 {
        vec![output, trailing]
    } else {
        vec![matrices, output, trailing]
    }
}

fn required(tensors: &CudaTensorSet, name: &str) -> Result<CudaTensor> {
    tensors.get(name).cloned().ok_or_else(|| Error::MissingTensor(name.into()))
}

fn shape(tensor: &CudaTensor, expected: Vec<usize>) -> Result<()> {
    if tensor.shape() != expected {
        return Err(Error::InvalidQuantizedTensor {
            name: tensor.name().into(),
            expected,
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
