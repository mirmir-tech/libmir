//! Expert banks from per-expert dense (BF16/F16/F32) projections, as
//! Transformers checkpoints store them (`experts.{e}.{gate,up,down}_proj`).

use models::weights::{BindingTransform, TensorBinding, TensorStorage};

use crate::engine::{Array, DenseLinear, Error, ModelTensors, Result, Stream};

/// Stacks each expert's `[output, input]` weight, and its bias where every
/// expert has one, into one `[experts, output, input]` bank that
/// [`DenseLinear::gather`] selects experts from. The bank is a copy, so the
/// per-expert tensors are taken from the model's files and freed once it is
/// built.
pub(in crate::engine::binding::projection) fn individual_bank(
    tensors: &ModelTensors,
    bindings: &[&TensorBinding],
    stream: &Stream,
) -> Result<DenseLinear> {
    let Some(first) = bindings.first() else {
        return Err(Error::InvalidModel("dense expert bank is empty".into()));
    };
    let mut weights = Vec::with_capacity(bindings.len());
    let mut biases = Vec::with_capacity(bindings.len());
    for binding in bindings {
        let TensorStorage::Dense { bias, .. } = &binding.storage else {
            return Err(invalid(binding, "expert storage differs within bank"));
        };
        let weight = tensors.take(&binding.source)?;
        let weight = if binding.transforms.contains(&BindingTransform::Transpose) {
            weight.transpose(&[1, 0], stream)?
        } else {
            weight
        };
        let shape = weight.shape()?;
        if shape.len() != 2 {
            return Err(invalid(binding, "individual expert is not an ordinary matrix"));
        }
        if weights
            .first()
            .map(Array::shape)
            .transpose()?
            .is_some_and(|first| first != shape)
        {
            return Err(invalid(binding, "expert matrix geometry differs within bank"));
        }
        weights.push(weight);
        biases.push(bias.as_deref().map(|name| tensors.take(name)).transpose()?);
    }
    let weight = stack(&weights.iter().collect::<Vec<_>>(), stream)?;
    let bias = match biases.iter().map(Option::as_ref).collect::<Option<Vec<_>>>() {
        Some(biases) => Some(stack(&biases, stream)?),
        None if biases.iter().all(Option::is_none) => None,
        None => return Err(invalid(first, "only some experts in the bank have a bias")),
    };
    drop(weights);
    // The reads freed into MLX's cache are not reused by the next bank's;
    // released now, a model's banks peak at one bank's reads, not all.
    crate::engine::clear_memory_cache()?;
    DenseLinear::from_binding_weight(weight, bias, false, stream)
}

fn stack(parts: &[&Array], stream: &Stream) -> Result<Array> {
    let parts = parts.iter().map(|part| part.native()).collect::<Vec<_>>();
    let stacked = Array::from_native(stream.native().graph().stack(&parts, 0)?)?;
    // Evaluated and cut from its graph, the bank no longer holds the
    // per-expert reads, which the files gave up to it: they free here, not
    // when the model's files close, so the bank costs the checkpoint once.
    stacked.async_eval(stream)?;
    stream.synchronize()?;
    stacked.detach_graph(stream)?;
    Ok(stacked)
}

fn invalid(binding: &TensorBinding, reason: &str) -> Error {
    Error::InvalidModel(format!("{}: {reason}", binding.source))
}
