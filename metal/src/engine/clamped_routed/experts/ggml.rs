//! gpt-oss experts from GGUF: GGML MXFP4 banks repacked once, at load, into
//! OCP nibbles, which read as words are MLX's MXFP4 layout, so the experts
//! run through MLX `gather_mxfp4` like the MXFP4 `SafeTensors` checkpoints.

use models::{
    gguf::GgmlType,
    weights::{TensorBinding, TensorStorage},
};

use super::{BoundLinear, ClampedRoutedConfig, ExpertLayout};
use crate::engine::{Array, Error, ModelTensors, Result, Stream};

/// Whether every routed projection is a GGML MXFP4 bank.
pub(super) fn is_ggml_mxfp4(bindings: [&TensorBinding; 3]) -> bool {
    bindings.iter().all(|binding| {
        matches!(&binding.storage, TensorStorage::Ggml { format, .. }
            if format.ggml_type == GgmlType::MXFP4)
    })
}

pub(super) fn ggml_layout(
    tensors: &ModelTensors,
    [gate, up, down]: [&TensorBinding; 3],
    config: ClampedRoutedConfig,
    stream: &Stream,
) -> Result<ExpertLayout> {
    let hidden = usize::try_from(config.hidden)?;
    let intermediate = usize::try_from(config.intermediate)?;
    let projection = |binding, columns| -> Result<BoundLinear> {
        let (blocks, scales, bias) = bank(tensors, binding, columns, stream)?;
        for array in [&blocks, &scales] {
            array.async_eval(stream)?;
        }
        BoundLinear::mxfp4_bank(&blocks, scales, bias, stream)
    };
    Ok(ExpertLayout::Separate {
        gate: projection(gate, hidden)?,
        up: projection(up, hidden)?,
        down: projection(down, intermediate)?,
    })
}

/// One `[experts, rows, columns]` bank as native `[experts, rows, columns /
/// 32, 16]` nibbles, `[experts, rows, columns / 32]` scales and its BF16 bias.
fn bank(
    tensors: &ModelTensors,
    binding: &TensorBinding,
    columns: usize,
    stream: &Stream,
) -> Result<(Array, Array, Array)> {
    let TensorStorage::Ggml { bias: Some(bias), .. } = &binding.storage else {
        return Err(invalid(binding, "requires a GGML MXFP4 bank with a bias"));
    };
    let experts = i32::try_from(experts(binding)?)?;
    let raw = tensors.get(&binding.source)?;
    let shape = raw.shape()?;
    let [_, rows, bytes] = *shape.as_slice() else {
        return Err(invalid(binding, "GGML expert bank is not [experts, rows, bytes]"));
    };
    if usize::try_from(bytes)? != columns / 32 * 17 {
        return Err(invalid(binding, "GGML expert rows differ from the routed width"));
    }
    let flat = raw.reshape(&[experts * rows, bytes], stream)?;
    let [blocks, scales] = stream.kernels().ggml_mxfp4_repack(stream, &flat)?;
    let blocks_per_row = i32::try_from(columns / 32)?;
    Ok((
        blocks.reshape(&[experts, rows, blocks_per_row, 16], stream)?,
        scales.reshape(&[experts, rows, blocks_per_row], stream)?,
        tensors.get(bias)?,
    ))
}

fn experts(binding: &TensorBinding) -> Result<usize> {
    binding
        .shape
        .first()
        .copied()
        .ok_or_else(|| invalid(binding, "GGML expert bank has no dimensions"))
}

fn invalid(binding: &TensorBinding, reason: &str) -> Error {
    Error::InvalidQuantization(format!("{}: {reason}", binding.source))
}
