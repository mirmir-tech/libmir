//! Checkpoint bytes a text-only load keeps resident.
//!
//! A checkpoint can ship components outside the decoder, such as a vision
//! tower without a supported vision contract, which the decoder contract
//! gives a top-level auxiliary role. Multi-token prediction modules load only
//! with speculative decoding. The runtime loads tensors only through the
//! roles it executes, so a text-only load leaves both on disk, and a
//! speculative load leaves the auxiliary components. A vision load keeps the
//! auxiliary components resident but still leaves the prediction modules
//! unless it speculates.

use models::{
    layout::{ModelLayout, VisionConfig},
    weights::{LogicalTensorRole, TensorCatalog, WeightBindingPlan},
};

/// Checkpoint tensors a load leaves on disk.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UnloadedTensors {
    /// Number of unloaded tensors.
    pub tensors: usize,
    /// Their payload bytes.
    pub bytes: u64,
    /// Of those, the bytes of multi-token prediction modules.
    pub prediction_bytes: u64,
}

/// Tensors outside the decoder in a text-only load, or `None` when every
/// checkpoint byte must be counted: no decoder contract. With a vision
/// contract, whose tower loads by name alongside the decoder, the auxiliary
/// components count as resident; the prediction modules still load only with
/// speculative decoding.
pub fn unloaded(
    catalog: &TensorCatalog,
    bindings: Option<&WeightBindingPlan>,
    vision: Option<&VisionConfig>,
) -> Option<UnloadedTensors> {
    bindings?
        .tensors
        .iter()
        .filter(|binding| match binding.role {
            LogicalTensorRole::Prediction { .. } => true,
            LogicalTensorRole::Auxiliary { .. } => vision.is_none(),
            LogicalTensorRole::Embedding
            | LogicalTensorRole::FinalNorm
            | LogicalTensorRole::Output
            | LogicalTensorRole::VisionProjection
            | LogicalTensorRole::Layer { .. } => false,
        })
        .flat_map(|binding| {
            let prediction = matches!(binding.role, LogicalTensorRole::Prediction { .. });
            binding.physical_sources().into_iter().map(move |name| (name, prediction))
        })
        .try_fold(UnloadedTensors::default(), |total, (name, prediction)| {
            let bytes = u64::try_from(catalog.get(name)?.payload_bytes().ok()?).ok()?;
            Some(UnloadedTensors {
                tensors: total.tensors + 1,
                bytes: total.bytes.checked_add(bytes)?,
                prediction_bytes: if prediction {
                    total.prediction_bytes.checked_add(bytes)?
                } else {
                    total.prediction_bytes
                },
            })
        })
}

/// Checkpoint file bytes minus the tensors a load leaves on disk: with
/// `speculation`, the prediction modules load.
pub fn weight_bytes(
    layout: &ModelLayout,
    unloaded: Option<&UnloadedTensors>,
    speculation: bool,
) -> u64 {
    let files = layout.weights.iter().map(|weight| weight.bytes).sum::<u64>();
    unloaded.map_or(files, |unloaded| {
        let loaded = if speculation {
            unloaded.prediction_bytes
        } else {
            0
        };
        files.saturating_sub(unloaded.bytes.saturating_sub(loaded))
    })
}
