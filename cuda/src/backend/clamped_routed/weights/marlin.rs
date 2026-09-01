use std::sync::Arc;

use mircuda::{DeviceBuffer, MarlinMxFp4RepackSpec};

use super::NativeExpertWeights;
use crate::{CudaBackend, CudaTensor, Error, Result};

#[derive(Debug)]
pub(in crate::backend::clamped_routed) struct MarlinMxFp4Bank {
    pub(in crate::backend::clamped_routed) weight: DeviceBuffer<u8>,
    pub(in crate::backend::clamped_routed) scales: DeviceBuffer<u8>,
    pub(in crate::backend::clamped_routed) input_features: usize,
    pub(in crate::backend::clamped_routed) output_features: usize,
}

#[derive(Debug)]
pub(in crate::backend::clamped_routed) struct MarlinMxFp4Banks {
    pub(in crate::backend::clamped_routed) gate_up: MarlinMxFp4Bank,
    pub(in crate::backend::clamped_routed) down: MarlinMxFp4Bank,
}

#[derive(Clone, Copy)]
struct MarlinMxFp4Geometry {
    padded_output: usize,
    padded_input: usize,
    output_features: usize,
    input_features: usize,
    layout: MarlinMxFp4Layout,
}

#[derive(Clone, Copy)]
enum MarlinMxFp4Layout {
    InterleavedGateUp,
    Sequential,
}

impl NativeExpertWeights {
    pub(in crate::backend::clamped_routed) fn marlin(
        &self,
        backend: &CudaBackend,
        experts: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Arc<MarlinMxFp4Banks>> {
        let mut cached = self
            .marlin
            .lock()
            .map_err(|_| Error::InvalidExecutionPlan("MXFP4 Marlin bank cache is poisoned"))?;
        if let Some(banks) = cached.as_ref() {
            return Ok(Arc::clone(banks));
        }
        let banks = Arc::new(MarlinMxFp4Banks {
            gate_up: prepare(
                backend,
                experts,
                MarlinMxFp4Geometry {
                    padded_output: round_up(intermediate, 128)?.checked_mul(2).ok_or(
                        Error::InvalidDecoderKernel("MXFP4 Marlin gate/up padding overflow"),
                    )?,
                    padded_input: round_up(hidden, 256)?,
                    output_features: intermediate.checked_mul(2).ok_or(
                        Error::InvalidDecoderKernel("MXFP4 Marlin gate/up width overflow"),
                    )?,
                    input_features: hidden,
                    layout: MarlinMxFp4Layout::InterleavedGateUp,
                },
                &self.gate_up_blocks,
                &self.gate_up_scales,
            )?,
            down: prepare(
                backend,
                experts,
                MarlinMxFp4Geometry {
                    padded_output: round_up(hidden, 256)?,
                    padded_input: round_up(intermediate, 128)?,
                    output_features: hidden,
                    input_features: intermediate,
                    layout: MarlinMxFp4Layout::Sequential,
                },
                &self.down_blocks,
                &self.down_scales,
            )?,
        });
        *cached = Some(Arc::clone(&banks));
        drop(cached);
        Ok(banks)
    }
}

fn prepare(
    backend: &CudaBackend,
    experts: usize,
    geometry: MarlinMxFp4Geometry,
    blocks: &CudaTensor,
    scales: &CudaTensor,
) -> Result<MarlinMxFp4Bank> {
    let matrix = experts
        .checked_mul(geometry.padded_output)
        .and_then(|value| value.checked_mul(geometry.padded_input))
        .ok_or(Error::InvalidDecoderKernel("MXFP4 Marlin bank size overflow"))?;
    let source_weight = u8s(blocks)?;
    let source_scales = u8s(scales)?;
    let logical_matrix = experts
        .checked_mul(geometry.output_features)
        .and_then(|value| value.checked_mul(geometry.input_features))
        .ok_or(Error::InvalidDecoderKernel("MXFP4 Marlin source size overflow"))?;
    if source_weight.len() != logical_matrix / 2 || source_scales.len() != logical_matrix / 32 {
        return Err(Error::InvalidDecoderKernel("invalid MXFP4 Marlin source bank extent"));
    }
    let mut weight = backend.pool().allocate(backend.stream(), matrix / 2)?;
    let mut prepared_scales = backend.pool().allocate(backend.stream(), matrix / 32)?;
    let spec = MarlinMxFp4RepackSpec::new(
        experts,
        geometry.padded_output,
        geometry.padded_input,
        geometry.output_features,
        geometry.input_features,
    )?;
    backend.context().marlin_repack_mxfp4(
        backend.stream(),
        spec,
        source_weight,
        &mut weight,
        geometry.layout.interleaved(),
    )?;
    backend.context().marlin_prepare_mxfp4_scales(
        backend.stream(),
        spec,
        source_scales,
        &mut prepared_scales,
        geometry.layout.interleaved(),
    )?;
    Ok(MarlinMxFp4Bank {
        weight,
        scales: prepared_scales,
        input_features: geometry.padded_input,
        output_features: geometry.padded_output,
    })
}

impl MarlinMxFp4Layout {
    const fn interleaved(self) -> bool {
        matches!(self, Self::InterleavedGateUp)
    }
}

fn round_up(value: usize, tile: usize) -> Result<usize> {
    value
        .checked_add(tile - 1)
        .map(|value| value / tile * tile)
        .ok_or(Error::InvalidDecoderKernel("MXFP4 Marlin padding overflow"))
}

fn u8s(tensor: &CudaTensor) -> Result<&DeviceBuffer<u8>> {
    tensor.as_u8().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "U8",
    })
}
