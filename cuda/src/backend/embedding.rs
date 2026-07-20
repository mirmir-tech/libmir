use mircuda::{DeviceBuffer, Stream, bf16};

use super::CudaBackend;
use crate::{
    CudaTensor, Error, Result,
    backend::{AffineQuantizedConfig, AffineQuantizedTensors},
    kernels::{AffineEmbedding, AffineEmbeddingSpec, Embedding},
};

/// Fixed-shape BF16 token embedding lookup.
#[derive(Clone, Debug)]
pub struct Bf16Embedding {
    operation: Embedding,
    stream: Stream,
    vocab: usize,
    hidden: usize,
}

/// Fixed-shape affine Int4/Int8 token embedding lookup.
#[derive(Clone, Debug)]
pub struct AffineQuantizedEmbedding {
    operation: AffineEmbedding,
    stream: Stream,
    config: AffineQuantizedConfig,
}

impl CudaBackend {
    pub fn prepare_bf16_embedding(
        &self,
        vocab: usize,
        hidden: usize,
        scale: f32,
    ) -> Result<Bf16Embedding> {
        Ok(Bf16Embedding {
            operation: Embedding::compile(&self.inner.compiler, vocab, hidden, scale)?,
            stream: self.inner.stream.clone(),
            vocab,
            hidden,
        })
    }

    pub fn prepare_affine_embedding(
        &self,
        config: AffineQuantizedConfig,
        scale: f32,
    ) -> Result<AffineQuantizedEmbedding> {
        Ok(AffineQuantizedEmbedding {
            operation: AffineEmbedding::compile(
                &self.inner.compiler,
                AffineEmbeddingSpec {
                    vocab: config.output_features,
                    hidden: config.input_features,
                    group_size: config.group_size,
                    bits: config.bits,
                    output_scale: scale,
                },
            )?,
            stream: self.inner.stream.clone(),
            config,
        })
    }
}

impl Bf16Embedding {
    pub fn execute(
        &self,
        selected: &DeviceBuffer<u32>,
        selected_index: usize,
        weight: &CudaTensor,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if weight.shape() != [self.vocab, self.hidden] {
            return Err(Error::InvalidQuantizedTensor {
                name: weight.name().into(),
                expected: vec![self.vocab, self.hidden],
                actual: weight.shape().to_vec(),
            });
        }
        let weight = weight.as_bf16().ok_or_else(|| Error::DTypeMismatch {
            name: weight.name().into(),
            expected: "BF16",
        })?;
        self.operation.execute(&self.stream, weight, selected, selected_index, output)
    }

    pub fn execute_batch(
        &self,
        selected: &DeviceBuffer<u32>,
        selected_start: usize,
        tokens: usize,
        weight: &CudaTensor,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if weight.shape() != [self.vocab, self.hidden] {
            return Err(Error::InvalidQuantizedTensor {
                name: weight.name().into(),
                expected: vec![self.vocab, self.hidden],
                actual: weight.shape().to_vec(),
            });
        }
        let weight = weight.as_bf16().ok_or_else(|| Error::DTypeMismatch {
            name: weight.name().into(),
            expected: "BF16",
        })?;
        self.operation
            .execute_batch(&self.stream, weight, selected, selected_start, tokens, output)
    }

    pub(crate) fn validate_token(&self, token: u32) -> Result<()> {
        if usize::try_from(token)? < self.vocab {
            Ok(())
        } else {
            Err(Error::InvalidToken { token, vocab: self.vocab })
        }
    }
}

impl AffineQuantizedEmbedding {
    pub fn execute_batch(
        &self,
        selected: &DeviceBuffer<u32>,
        selected_start: usize,
        tokens: usize,
        tensors: AffineQuantizedTensors<'_>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let packed = self.config.input_features / (32 / self.config.bits);
        let groups = self.config.input_features / self.config.group_size;
        validate(tensors.weight, &[self.config.output_features, packed], "U32")?;
        validate(tensors.scales, &[self.config.output_features, groups], "BF16")?;
        validate(tensors.biases, &[self.config.output_features, groups], "BF16")?;
        self.operation.execute(
            &self.stream,
            tensors.weight.as_u32().ok_or_else(|| Error::DTypeMismatch {
                name: tensors.weight.name().into(),
                expected: "U32",
            })?,
            tensors.scales.as_bf16().ok_or_else(|| Error::DTypeMismatch {
                name: tensors.scales.name().into(),
                expected: "BF16",
            })?,
            tensors.biases.as_bf16().ok_or_else(|| Error::DTypeMismatch {
                name: tensors.biases.name().into(),
                expected: "BF16",
            })?,
            selected,
            selected_start,
            tokens,
            output,
        )
    }

    pub fn validate_token(&self, token: u32) -> Result<()> {
        if usize::try_from(token)? < self.config.output_features {
            Ok(())
        } else {
            Err(Error::InvalidToken {
                token,
                vocab: self.config.output_features,
            })
        }
    }
}

fn validate(tensor: &CudaTensor, shape: &[usize], dtype: &'static str) -> Result<()> {
    if tensor.shape() != shape {
        return Err(Error::InvalidQuantizedTensor {
            name: tensor.name().into(),
            expected: shape.to_vec(),
            actual: tensor.shape().to_vec(),
        });
    }
    let matches = match dtype {
        "U32" => tensor.as_u32().is_some(),
        "BF16" => tensor.as_bf16().is_some(),
        _ => false,
    };
    if matches {
        Ok(())
    } else {
        Err(Error::DTypeMismatch {
            name: tensor.name().into(),
            expected: dtype,
        })
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use mircuda::{DeviceElement, bf16};

    use super::*;
    use crate::CudaConfig;

    #[test]
    fn dequantizes_selected_affine_embedding_rows() -> Result<()> {
        let backend = CudaBackend::new(CudaConfig::default())?;
        let operation = AffineEmbedding::compile(
            &backend.inner.compiler,
            AffineEmbeddingSpec {
                vocab: 2,
                hidden: 8,
                group_size: 4,
                bits: 4,
                output_scale: 1.0,
            },
        )?;
        let weight = copy(&backend, &[0_u32, 0x7654_3210])?;
        let scales = copy(&backend, &bf16s(&[1.0, 1.0, 1.0, 2.0]))?;
        let biases = copy(&backend, &bf16s(&[0.0, 0.0, 0.0, 10.0]))?;
        let selected = copy(&backend, &[1_u32])?;
        let mut output = backend.inner.pool.allocate(&backend.inner.stream, 8)?;
        operation.execute(
            &backend.inner.stream,
            &weight,
            &scales,
            &biases,
            &selected,
            0,
            1,
            &mut output,
        )?;
        let actual =
            read(&backend, &output)?.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
        assert_eq!(actual, [0.0, 1.0, 2.0, 3.0, 18.0, 20.0, 22.0, 24.0]);
        Ok(())
    }

    fn bf16s(values: &[f32]) -> Vec<bf16> {
        values.iter().copied().map(bf16::from_f32).collect()
    }

    fn copy<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<DeviceBuffer<T>> {
        let mut host = backend.inner.context.allocate_pinned(values.len())?;
        host.copy_from_slice(values)?;
        let mut device = backend.inner.pool.allocate(&backend.inner.stream, values.len())?;
        backend.inner.stream.copy_to_device(&mut host, &mut device)?;
        Ok(device)
    }

    fn read<T: DeviceElement>(backend: &CudaBackend, source: &DeviceBuffer<T>) -> Result<Vec<T>> {
        let mut host = backend.inner.context.allocate_pinned(source.len())?;
        backend.inner.stream.copy_to_host(source, &mut host)?;
        Ok(host.to_vec()?)
    }
}
