use mircuda::{DenseMatmulPlan, DenseMatmulSpec, DeviceBuffer, PinnedBuffer, f16};

use super::{CudaSequenceScoringModel, scratch::Scratch};
use crate::{
    CudaTensor, Error, Result,
    kernels::{EncoderElementwiseF16, EncoderElementwiseSpec},
};

impl CudaSequenceScoringModel {
    pub fn score(&self, tokens: &[u32]) -> Result<f32> {
        if tokens.is_empty() {
            return Err(Error::InvalidDecoderKernel("empty CUDA scoring pair"));
        }
        if let Some(&token) = tokens.iter().find(|&&token| {
            usize::try_from(token).map_or(true, |token| token >= self.config.vocab_size)
        }) {
            return Err(Error::InvalidToken { token, vocab: self.config.vocab_size });
        }
        let backend = &self.backend;
        let config = &self.config;
        let count = tokens.len();
        let mut ids_host: PinnedBuffer<u32> = backend.inner.context.allocate_pinned(count)?;
        ids_host.copy_from_slice(tokens)?;
        let mut ids = backend.inner.pool.allocate::<u32>(&backend.inner.stream, count)?;
        backend.inner.stream.copy_to_device(&mut ids_host, &mut ids)?;
        let mut scratch =
            Scratch::new(backend, count, config.hidden_size, config.intermediate_size)?;
        ops(backend, count, config.hidden_size, epsilon(config)?)?.embedding_norm(
            &backend.inner.stream,
            &ids,
            f16_tensor(self.tensor("new.embeddings.word_embeddings.weight")?)?,
            f16_tensor(self.tensor("new.embeddings.token_type_embeddings.weight")?)?,
            f16_tensor(self.tensor("new.embeddings.LayerNorm.weight")?)?,
            f16_tensor(self.tensor("new.embeddings.LayerNorm.bias")?)?,
            &mut scratch.first,
        )?;
        for layer in 0..config.num_hidden_layers {
            self.layer(layer, &mut scratch)?;
        }
        let hidden = if config.num_hidden_layers.is_multiple_of(2) {
            &scratch.first
        } else {
            &scratch.second
        };
        backend.inner.stream.copy_device_range(
            hidden,
            0..config.hidden_size,
            &mut scratch.cls,
            0,
        )?;
        project(
            backend,
            1,
            config.hidden_size,
            config.hidden_size,
            &scratch.cls,
            self.tensor("new.pooler.dense.weight")?,
            &mut scratch.pooled_raw,
        )?;
        ops(backend, 1, config.hidden_size, epsilon(config)?)?.tanh_bias(
            &backend.inner.stream,
            &scratch.pooled_raw,
            f16_tensor(self.tensor("new.pooler.dense.bias")?)?,
            &mut scratch.pooled,
        )?;
        project(
            backend,
            1,
            config.hidden_size,
            1,
            &scratch.pooled,
            self.tensor("classifier.weight")?,
            &mut scratch.score_raw,
        )?;
        ops(backend, 1, 1, epsilon(config)?)?.add_bias(
            &backend.inner.stream,
            &scratch.score_raw,
            f16_tensor(self.tensor("classifier.bias")?)?,
            &mut scratch.score,
        )?;
        let mut host: PinnedBuffer<f16> = backend.inner.context.allocate_pinned(1)?;
        backend.inner.stream.copy_to_host(&scratch.score, &mut host)?;
        Ok(host.to_vec()?[0].to_f32())
    }

    pub(super) fn tensor(&self, name: &str) -> Result<&CudaTensor> {
        self.tensors.get(name).ok_or_else(|| Error::MissingTensor(name.into()))
    }
}

pub(super) fn project(
    backend: &crate::CudaBackend,
    rows: usize,
    input_width: usize,
    output_width: usize,
    input: &DeviceBuffer<f16>,
    weight: &CudaTensor,
    output: &mut DeviceBuffer<f16>,
) -> Result<()> {
    let weight = f16_tensor(weight)?;
    let mut plan = DenseMatmulPlan::<f16>::new(
        &backend.inner.context,
        &backend.inner.stream,
        DenseMatmulSpec::new(rows, output_width, input_width)?,
    )?;
    Ok(plan.execute(&backend.inner.stream, input, weight, output, 1.0, 0.0)?)
}

pub(super) fn ops(
    backend: &crate::CudaBackend,
    rows: usize,
    columns: usize,
    epsilon: f32,
) -> Result<EncoderElementwiseF16> {
    EncoderElementwiseF16::compile(
        &backend.inner.compiler,
        EncoderElementwiseSpec { rows, columns, epsilon },
    )
}

pub(super) fn f16_tensor(tensor: &CudaTensor) -> Result<&DeviceBuffer<f16>> {
    tensor.as_f16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "F16",
    })
}

pub(super) fn epsilon(config: &models::layout::EncoderConfig) -> Result<f32> {
    Ok(config.layer_norm_eps.to_string().parse()?)
}

pub(super) fn buffer(backend: &crate::CudaBackend, elements: usize) -> Result<DeviceBuffer<f16>> {
    Ok(backend.inner.pool.allocate::<f16>(&backend.inner.stream, elements)?)
}
