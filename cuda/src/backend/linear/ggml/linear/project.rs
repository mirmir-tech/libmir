//! One projection of a prepared GGML plan, before its output scale.

use mircuda::{DeviceBuffer, bf16};

use super::{GgmlBf16Linear, GgmlCheckpointWeight, Path};
use crate::{
    CudaTensor, Error, Result,
    kernels::{GgmlBankRows, GgmlMmqGroups, GgmlMmqInput, GgmlMmqPlacement},
};

impl GgmlBf16Linear {
    pub(super) fn project(
        &self,
        input: &DeviceBuffer<bf16>,
        weight: &GgmlCheckpointWeight,
        output: &mut DeviceBuffer<bf16>,
        quantized: bool,
    ) -> Result<()> {
        let missing = || Error::InvalidExecutionPlan("GGML plan has no kernel for the weight type");
        match &self.path {
            Path::Vector { operations, activations } => {
                let operation = operations
                    .iter()
                    .find(|operation| operation.spec().kind == weight.kind)
                    .ok_or_else(missing)?;
                let mut activations = activations
                    .lock()
                    .map_err(|_| Error::InvalidExecutionPlan("GGML activation lock is poisoned"))?;
                if !quantized {
                    operation.quantize(&self.stream, input, &mut activations)?;
                }
                operation.multiply(&self.stream, &activations, &weight.blocks, output)
            },
            Path::Grouped {
                operation,
                activations,
                offsets,
                counts,
                tokens,
            } => {
                let poisoned = || Error::InvalidExecutionPlan("GGML grouped plan lock is poisoned");
                let mut operation = operation.lock().map_err(|_| poisoned())?;
                let mut activations = activations.lock().map_err(|_| poisoned())?;
                let groups = GgmlMmqGroups { offsets, counts };
                // Each plan lays out its own tiles, so it quantizes the input
                // again even when an earlier projection of it did.
                operation.prepare(
                    &self.stream,
                    GgmlMmqInput::Rows(input),
                    *tokens,
                    groups,
                    &mut activations,
                )?;
                let placement = GgmlMmqPlacement {
                    ld: self.output_features,
                    stride: 1,
                    offset: 0,
                };
                let bank = (&weight.blocks, GgmlBankRows::whole(self.output_features));
                operation.multiply(
                    &self.stream, &activations, groups, weight.kind, bank, output, placement,
                )
            },
            Path::Dense { dequantize, projection, scratch } => {
                let dequantize = dequantize
                    .iter()
                    .find(|operation| operation.kind() == weight.kind)
                    .ok_or_else(missing)?;
                let poisoned = || Error::InvalidExecutionPlan("GGML scratch lock is poisoned");
                // The stream orders this projection's use of the scratch
                // before the next one's; the lock only guards the handle.
                let buffer = {
                    let mut scratch = scratch.lock().map_err(|_| poisoned())?;
                    dequantize.execute(&self.stream, &weight.blocks, &mut scratch)?;
                    scratch.clone()
                };
                let dense = CudaTensor::from_bf16(
                    "GGML dequantized projection".into(),
                    vec![self.output_features, self.input_features],
                    buffer,
                );
                projection.lock().map_err(|_| poisoned())?.execute(input, &dense, output)
            },
        }
    }
}
