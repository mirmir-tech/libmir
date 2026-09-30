use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};

use mircup::{DType, EmbeddingTable, LayerNorm, Linear, Tensor};
use models::decision::{DecisionTensor, DecisionTensorPlan};

use crate::{Error, Result};

/// Reads planned tensors from their safetensors payloads.
pub struct WeightReader<'a> {
    plan: &'a DecisionTensorPlan,
    eps: f32,
}

impl<'a> WeightReader<'a> {
    pub const fn new(plan: &'a DecisionTensorPlan, eps: f32) -> Self {
        Self { plan, eps }
    }

    pub fn tensor(&self, role: DecisionTensor) -> Result<Tensor> {
        let (dtype, shape, bytes) = self.payload(role)?;
        Ok(Tensor::from_le_bytes(dtype, shape, &bytes)?)
    }

    pub fn values(&self, role: DecisionTensor) -> Result<Vec<f32>> {
        Ok(self.tensor(role)?.into_data())
    }

    pub fn linear(&self, weight: DecisionTensor, bias: Option<DecisionTensor>) -> Result<Linear> {
        let bias = bias.map(|bias| self.values(bias)).transpose()?;
        Ok(Linear::new(self.tensor(weight)?, bias)?)
    }

    /// A norm with the encoder epsilon and no bias.
    pub fn encoder_norm(&self, weight: DecisionTensor) -> Result<LayerNorm> {
        Ok(LayerNorm::new(self.values(weight)?, None, self.eps)?)
    }

    /// A `PyTorch` `nn.LayerNorm` with its default epsilon.
    pub fn biased_norm(&self, weight: DecisionTensor, bias: DecisionTensor) -> Result<LayerNorm> {
        Ok(LayerNorm::new(self.values(weight)?, Some(self.values(bias)?), 1e-5)?)
    }

    pub fn embedding(&self, role: DecisionTensor) -> Result<EmbeddingTable> {
        let (dtype, shape, bytes) = self.payload(role)?;
        let &[rows, width] = shape.as_slice() else {
            return Err(Error::NotMatrix(role.name()));
        };
        Ok(EmbeddingTable::from_le_bytes(dtype, rows, width, bytes)?)
    }

    fn payload(&self, role: DecisionTensor) -> Result<(DType, Vec<usize>, Vec<u8>)> {
        let info = self.plan.get(role)?;
        let dtype = match info.dtype.as_str() {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            other => {
                return Err(Error::DType {
                    name: info.name.clone(),
                    dtype: other.into(),
                });
            },
        };
        let mut file = File::open(&info.file)?;
        let _position = file.seek(SeekFrom::Start(info.payload_start()?))?;
        let mut bytes = vec![0; info.payload_bytes()?];
        file.read_exact(&mut bytes)?;
        Ok((dtype, info.shape.clone(), bytes))
    }
}
