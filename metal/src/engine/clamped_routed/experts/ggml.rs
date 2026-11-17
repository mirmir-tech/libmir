//! gpt-oss experts from GGUF: GGML MXFP4 banks repacked once, at load, into
//! the native layout of the MXFP4 expert kernels; gate and up rows are
//! interleaved as the fused native bank stores them.

use models::{
    gguf::GgmlType,
    weights::{TensorBinding, TensorStorage},
};

use super::{ClampedRoutedConfig, ExpertLayout, native::validate};
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
    let (gate_blocks, gate_scales, gate_bias) = bank(tensors, gate, hidden, stream)?;
    let (up_blocks, up_scales, up_bias) = bank(tensors, up, hidden, stream)?;
    let (down_blocks, down_scales, down_bias) = bank(tensors, down, intermediate, stream)?;
    let experts = experts(gate)?;
    let layout = ExpertLayout::Native {
        gate_up_blocks: interleave(&gate_blocks, &up_blocks, stream)?,
        gate_up_scales: interleave(&gate_scales, &up_scales, stream)?,
        gate_up_bias: interleave(&gate_bias, &up_bias, stream)?,
        down_blocks,
        down_scales,
        down_bias,
    };
    validate(&layout, experts, hidden, intermediate, 2 * intermediate, gate, down)?;
    if let ExpertLayout::Native {
        gate_up_blocks,
        gate_up_scales,
        gate_up_bias,
        ..
    } = &layout
    {
        for array in [gate_up_blocks, gate_up_scales, gate_up_bias] {
            array.async_eval(stream)?;
        }
    }
    Ok(layout)
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

/// `[experts, rows, ..]` gate and up as `[experts, 2 rows, ..]`, row 2j from
/// gate row j and row 2j + 1 from up row j.
fn interleave(gate: &Array, up: &Array, stream: &Stream) -> Result<Array> {
    let shape = gate.shape()?;
    let mut split = shape.clone();
    split.insert(2, 1);
    let joined = Array::concatenate(
        &[&gate.reshape(&split, stream)?, &up.reshape(&split, stream)?],
        2,
        stream,
    )?;
    let mut fused = shape;
    fused[1] *= 2;
    joined.reshape(&fused, stream)
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
