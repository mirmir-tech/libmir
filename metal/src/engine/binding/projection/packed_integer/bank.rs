//! Expert banks from per-expert compressed-tensors projections, as GSQ
//! checkpoints store them (`experts.{e}.{gate,up,down}_proj`).

use models::weights::TensorBinding;

use super::{arrays, invalid};
use crate::engine::{Array, Error, QuantizedArrays, QuantizedLinear, Result, Stream};

/// Stacks each expert's affine weight, scales and biases into one
/// `[experts, output, ...]` bank for MLX's `gather_qmm`.
pub(in crate::engine::binding::projection) fn individual_bank(
    tensors: &crate::engine::ModelTensors,
    bindings: &[&TensorBinding],
    stream: &Stream,
) -> Result<QuantizedLinear> {
    let Some(first) = bindings.first() else {
        return Err(Error::InvalidQuantization("packed-integer expert bank is empty".into()));
    };
    let experts = bindings
        .iter()
        .map(|binding| {
            if binding.logical_shape != first.logical_shape {
                return Err(invalid(binding, "expert matrix geometry differs within bank"));
            }
            arrays(tensors, binding, stream)
        })
        .collect::<Result<Vec<_>>>()?;
    let (_, group_size, bits) = experts[0];
    if experts.iter().any(|(_, group, width)| (*group, *width) != (group_size, bits)) {
        return Err(invalid(first, "expert quantization differs within bank"));
    }
    let stack = |part: fn(&QuantizedArrays) -> &Array| -> Result<Array> {
        let parts = experts.iter().map(|(arrays, ..)| part(arrays).native()).collect::<Vec<_>>();
        let stacked = Array::from_native(stream.native().graph().stack(&parts, 0)?)?;
        stacked.async_eval(stream)?;
        Ok(stacked)
    };
    let weight = stack(|arrays| &arrays.weight)?;
    let scales = stack(|arrays| &arrays.scales)?;
    let biases = stack(|arrays| &arrays.biases)?;
    Ok(QuantizedLinear::from_quantized(
        QuantizedArrays::new(weight, scales, biases, group_size, bits)?,
        group_size,
        bits,
    ))
}
