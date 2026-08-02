use mircuda::{DeviceBuffer, PinnedBuffer, bf16};

use super::{CudaTextEmbeddingModel, scratch::Scratch};
use crate::{
    CudaTensor, Error, Result,
    kernels::{L2NormalizeBf16, RopeSpec, SelectRowBf16},
};

impl CudaTextEmbeddingModel {
    pub fn embed(&self, tokens: &[u32], dimensions: usize) -> Result<Vec<f32>> {
        let config = &self.config;
        if tokens.is_empty() || dimensions == 0 || dimensions > config.hidden_size {
            return Err(Error::InvalidDecoderKernel("invalid CUDA embedding request"));
        }
        let count = tokens.len();
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let query = config.num_attention_heads * config.head_dim;
        let key_value = config.num_key_value_heads * config.head_dim;
        let backend = &self.backend;
        let stream = &backend.inner.stream;
        let mut ids_host: PinnedBuffer<u32> = backend.inner.context.allocate_pinned(count)?;
        ids_host.copy_from_slice(tokens)?;
        let mut ids = backend.inner.pool.allocate::<u32>(stream, count)?;
        stream.copy_to_device(&mut ids_host, &mut ids)?;
        let mut first = buffer(backend, count * hidden)?;
        let mut second = buffer(backend, count * hidden)?;
        backend.prepare_bf16_embedding(config.vocab_size, hidden, 1.0)?.execute_batch(
            &ids,
            0,
            count,
            self.tensor(&self.layout.name("embed_tokens.weight"))?,
            &mut first,
        )?;
        let mut scratch = Scratch::new(backend, count, hidden, intermediate, query, key_value)?;
        for layer in 0..config.num_hidden_layers {
            let (input, output) = if layer.is_multiple_of(2) {
                (&first, &mut second)
            } else {
                (&second, &mut first)
            };
            self.layer(layer, input, output, &mut scratch)?;
        }
        let input = if config.num_hidden_layers.is_multiple_of(2) {
            &first
        } else {
            &second
        };
        backend.prepare_rms_norm_bf16(count, hidden, epsilon(config)?)?.execute(
            input,
            self.tensor(&self.layout.name("norm.weight"))?,
            &mut scratch.normalized,
        )?;
        SelectRowBf16::compile(&backend.inner.compiler, hidden)?.execute(
            stream,
            &scratch.normalized,
            &mut scratch.selected,
            count - 1,
            count,
        )?;
        let mut prefix = buffer(backend, dimensions)?;
        let mut embedding = buffer(backend, dimensions)?;
        stream.copy_device_range(&scratch.selected, 0..dimensions, &mut prefix, 0)?;
        L2NormalizeBf16::compile(&backend.inner.compiler, dimensions)?
            .execute(stream, &prefix, &mut embedding)?;
        backend.read_logits(&embedding)
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
    input: &DeviceBuffer<bf16>,
    weight: &CudaTensor,
    output: &mut DeviceBuffer<bf16>,
) -> Result<()> {
    backend
        .prepare_bf16_linear(rows, input_width, output_width)?
        .execute(input, weight, output)
}

pub(super) fn rope(
    backend: &crate::CudaBackend,
    tokens: usize,
    heads: usize,
    head: usize,
    theta: f32,
) -> Result<crate::RopeBf16> {
    backend.prepare_rope_bf16(RopeSpec {
        tokens,
        heads,
        head_dim: head,
        rotary_dim: head,
        pairing_dim: head,
        theta,
    })
}

pub(super) fn epsilon(config: &models::layout::DecoderConfig) -> Result<f32> {
    Ok(config.rms_norm_eps.to_string().parse()?)
}

pub(super) fn theta(config: &models::layout::DecoderConfig) -> Result<f32> {
    Ok(config.rope_theta.unwrap_or(10_000.0).to_string().parse()?)
}

pub(super) fn buffer(backend: &crate::CudaBackend, elements: usize) -> Result<DeviceBuffer<bf16>> {
    Ok(backend.inner.pool.allocate::<bf16>(&backend.inner.stream, elements)?)
}
