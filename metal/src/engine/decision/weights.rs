use models::decision::{DecisionTensor, DecisionTensorPlan};

use crate::engine::{Array, Dtype, Error, Result, Stream, TensorFile};

/// Reads planned Laya tensors and keeps compute weights in `f32`, matching
/// the fp32 reference runtime.
pub struct WeightReader<'a> {
    plan: &'a DecisionTensorPlan,
    file: TensorFile,
    stream: &'a Stream,
}

impl<'a> WeightReader<'a> {
    pub fn open(plan: &'a DecisionTensorPlan, stream: &'a Stream) -> Result<Self> {
        let path = plan.get(DecisionTensor::TokenEmbedding)?.file.clone();
        if plan.iter().any(|(_, tensor)| tensor.file != path) {
            return Err(Error::InvalidModel(
                "Laya weights must live in one safetensors file".into(),
            ));
        }
        // MLX reads safetensors payloads only on a CPU stream; the GPU stream
        // converts and uses them.
        Ok(Self {
            plan,
            file: TensorFile::load(&path, &Stream::new_cpu()?)?,
            stream,
        })
    }

    /// The tensor in its checkpoint encoding.
    pub fn stored(&self, role: DecisionTensor) -> Result<Array> {
        self.file.get(&self.plan.get(role)?.name)
    }

    pub fn float(&self, role: DecisionTensor) -> Result<Array> {
        self.stored(role)?.astype(Dtype::Float32, self.stream)
    }

    pub fn linear(&self, weight: DecisionTensor, bias: Option<DecisionTensor>) -> Result<Linear> {
        Ok(Linear {
            transposed: self.float(weight)?.transpose(&[1, 0], self.stream)?,
            bias: bias.map(|bias| self.float(bias)).transpose()?,
        })
    }

    /// A norm without bias; `LayerNorm` with a zero bias is the same map.
    pub fn unbiased_norm(&self, weight: DecisionTensor, eps: f32) -> Result<Norm> {
        let weight = self.float(weight)?;
        let width = weight.shape()?;
        let zeros = vec![0.0; usize::try_from(width.iter().product::<i32>())?];
        Ok(Norm {
            bias: Array::from_f32(&zeros, &width)?,
            weight,
            eps,
        })
    }

    /// A `PyTorch` `nn.LayerNorm` with its default epsilon.
    pub fn biased_norm(&self, weight: DecisionTensor, bias: DecisionTensor) -> Result<Norm> {
        Ok(Norm {
            weight: self.float(weight)?,
            bias: self.float(bias)?,
            eps: 1e-5,
        })
    }
}

/// `y = x Wᵀ + b` with the transposed weight kept resident.
pub struct Linear {
    transposed: Array,
    bias: Option<Array>,
}

impl Linear {
    pub fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let output = input.matmul(&self.transposed, stream)?;
        match &self.bias {
            Some(bias) => output.add(bias, stream),
            None => Ok(output),
        }
    }
}

pub struct Norm {
    weight: Array,
    bias: Array,
    eps: f32,
}

impl Norm {
    pub fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        input.layer_norm(&self.weight, &self.bias, self.eps, stream)
    }
}
