//! Checkpoint bytes a text-only load keeps resident.
//!
//! A checkpoint can ship components outside the decoder, such as a vision
//! tower without a supported vision contract, which the decoder contract
//! gives a top-level auxiliary role. Multi-token prediction modules have
//! typed roles but load only with speculative decoding, which no backend runs
//! yet. The runtime loads tensors only through the roles it executes, so a
//! text-only load leaves both on disk.

use models::{
    execution::DecoderExecutionContract,
    layout::{ModelLayout, VisionConfig},
    weights::{LogicalTensorRole, TensorCatalog},
};

/// Checkpoint tensors a load leaves on disk.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UnloadedTensors {
    /// Number of unloaded tensors.
    pub tensors: usize,
    /// Their payload bytes.
    pub bytes: u64,
}

/// Tensors outside the decoder in a text-only load, or `None` when every
/// checkpoint byte must be counted: no decoder contract, or a vision contract
/// whose tower loads by name alongside the decoder.
pub fn unloaded(
    catalog: &TensorCatalog,
    execution: Option<&DecoderExecutionContract>,
    vision: Option<&VisionConfig>,
) -> Option<UnloadedTensors> {
    if vision.is_some() {
        return None;
    }
    execution?
        .bindings
        .tensors
        .iter()
        .filter(|binding| {
            matches!(
                binding.role,
                LogicalTensorRole::Auxiliary { .. } | LogicalTensorRole::Prediction { .. }
            )
        })
        .flat_map(|binding| binding.physical_sources())
        .try_fold(UnloadedTensors::default(), |total, name| {
            let bytes = u64::try_from(catalog.get(name)?.payload_bytes().ok()?).ok()?;
            Some(UnloadedTensors {
                tensors: total.tensors + 1,
                bytes: total.bytes.checked_add(bytes)?,
            })
        })
}

/// Checkpoint file bytes minus the tensors a text-only load leaves on disk.
pub fn weight_bytes(layout: &ModelLayout, unloaded: Option<&UnloadedTensors>) -> u64 {
    let files = layout.weights.iter().map(|weight| weight.bytes).sum::<u64>();
    unloaded.map_or(files, |unloaded| files.saturating_sub(unloaded.bytes))
}
