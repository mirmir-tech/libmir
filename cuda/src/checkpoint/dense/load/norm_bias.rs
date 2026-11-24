//! `LayerNorm` biases (Phi-3.5-MoE). The biased norm kernel reads a norm's
//! bias after its weight in one buffer, so each biased stream norm is packed
//! at load and the layer normalizes with
//! [`NormFormula::MeanCentredBiased`].

use models::weights::TensorBinding;

use crate::{
    CudaBackend, CudaTensor, CudaTensorSet, DenseSwiGluConfig, Error, Result,
    checkpoint::dense::DenseLayerBindings, kernels::NormFormula,
};

/// Packs the biases of the layer's stream norms and selects the biased
/// formula; Q/K and sandwich norms take no bias.
pub(super) fn pack(
    backend: &CudaBackend,
    tensors: &mut CudaTensorSet,
    bindings: DenseLayerBindings<'_>,
    block: &mut DenseSwiGluConfig,
) -> Result<()> {
    let stream_norms = stream_norms(bindings);
    let biased = stream_norms.iter().filter(|norm| norm.dense_bias().is_some()).count();
    let others = bindings
        .norms()
        .into_iter()
        .filter(|norm| !stream_norms.iter().any(|stream| std::ptr::eq(*stream, *norm)));
    if others.into_iter().any(|norm| norm.dense_bias().is_some()) {
        return Err(unsupported("dense CUDA Q/K and sandwich norms have no bias"));
    }
    if biased == 0 {
        return Ok(());
    }
    if biased != stream_norms.len() || block.attention.stream_norm != NormFormula::MeanCentred {
        return Err(unsupported("dense CUDA biases every LayerNorm of a layer or none"));
    }
    for norm in stream_norms {
        append_bias(backend, tensors, norm)?;
    }
    block.attention.stream_norm = NormFormula::MeanCentredBiased;
    Ok(())
}

/// Replaces the norm `binding` names with its weight followed by its bias.
pub(in crate::checkpoint) fn append_bias(
    backend: &CudaBackend,
    tensors: &mut CudaTensorSet,
    binding: &TensorBinding,
) -> Result<()> {
    let bias = binding
        .dense_bias()
        .ok_or_else(|| Error::MissingTensor(binding.source.clone()))?;
    let bf16 = |name: &str| {
        tensors
            .get(name)
            .and_then(CudaTensor::as_bf16)
            .ok_or_else(|| Error::DTypeMismatch { name: name.into(), expected: "BF16" })
    };
    let (weight, bias) = (bf16(&binding.source)?, bf16(bias)?);
    let columns = weight.len();
    if bias.len() != columns {
        return Err(unsupported("a LayerNorm bias differs from its weight in width"));
    }
    let stream = backend.stream();
    let mut packed = backend.pool().allocate(stream, columns * 2)?;
    stream.copy_device_range(weight, 0..columns, &mut packed, 0)?;
    stream.copy_device_range(bias, 0..columns, &mut packed, columns)?;
    tensors.replace(CudaTensor::from_bf16(binding.source.clone(), vec![columns * 2], packed))
}

/// The input and post-attention norms, once each (a parallel layer binds
/// one norm as both).
fn stream_norms(bindings: DenseLayerBindings<'_>) -> Vec<&TensorBinding> {
    let mut norms: Vec<_> =
        bindings.input_norm.into_iter().chain(bindings.post_attention_norm).collect();
    norms.dedup_by(|first, second| std::ptr::eq(*first, *second));
    norms
}

fn unsupported(message: &str) -> Error {
    Error::UnsupportedDecoderLayer(message.into())
}
