//! gpt-oss from GGUF: GGML projections for attention, a dense router, and
//! GGML MXFP4 expert banks repacked on the device to the native interleaved
//! OCP layout, so the routed experts run the native (Marlin) kernels.

use std::sync::{Arc, Mutex};

use mircuda::{DeviceBuffer, bf16};
use models::{
    gguf::GgmlType,
    weights::{RoutedDecoderLayerBindings, RoutedExpertBindings, TensorBinding},
};

use super::{
    ClampedRoutedConfig, ClampedRoutedExpertWeights, NativeExpertWeights, binding_bias, tensor,
};
use crate::{
    CudaBackend, CudaTensor, CudaTensorSet, Error, GgmlCheckpointWeight, Result,
    backend::clamped_routed::{
        projection::{ClampedRoutedLinearWeight, ClampedRoutedQkvProjections},
        validation::validate_native_experts,
    },
    kernels::{GgmlMxFp4Repack, GgmlRowPlacement},
};

pub(super) fn load(
    backend: &CudaBackend,
    config: ClampedRoutedConfig,
    tensors: &CudaTensorSet,
    bindings: RoutedDecoderLayerBindings<'_>,
) -> Result<(
    ClampedRoutedQkvProjections,
    ClampedRoutedLinearWeight,
    ClampedRoutedLinearWeight,
    ClampedRoutedExpertWeights,
)> {
    let weight = |binding| GgmlCheckpointWeight::load_binding(tensors, binding);
    let qkv = [weight(bindings.query)?, weight(bindings.key)?, weight(bindings.value)?];
    let output = weight(bindings.attention_output)?;
    let RoutedExpertBindings::SeparateGateUp { gate, up, down } = bindings.experts else {
        return Err(Error::InvalidDecoderKernel(
            "GGML clamped-routed requires separate gate, up and down expert banks",
        ));
    };
    let experts = experts(backend, config, tensors, [gate, up, down])?;
    validate_native_experts(config, &experts)?;
    Ok((
        ClampedRoutedQkvProjections::Ggml(Box::new(qkv)),
        ClampedRoutedLinearWeight::Ggml(output),
        ClampedRoutedLinearWeight::Native(tensor(tensors, &bindings.router.source)?),
        experts,
    ))
}

fn experts(
    backend: &CudaBackend,
    config: ClampedRoutedConfig,
    tensors: &CudaTensorSet,
    [gate, up, down]: [&TensorBinding; 3],
) -> Result<ClampedRoutedExpertWeights> {
    let repack = GgmlMxFp4Repack::compile(&backend.inner.compiler)?;
    let (experts, hidden, intermediate) = (config.experts, config.hidden, config.intermediate);
    let fused = Bank::new(backend, "gate_up", experts * 2 * intermediate, hidden)?;
    let separate = GgmlRowPlacement { stride: 1, offset: 0 };
    let mut fused = fused;
    for (binding, offset) in [(gate, 0), (up, 1)] {
        fused.fill(backend, &repack, tensors, binding, GgmlRowPlacement { stride: 2, offset })?;
    }
    let mut down_bank = Bank::new(backend, "down", experts * hidden, intermediate)?;
    down_bank.fill(backend, &repack, tensors, down, separate)?;
    let (gate_up_blocks, gate_up_scales, gate_up_bias) =
        fused.into_tensors(gate, [experts, 2 * intermediate, hidden / 32]);
    let (down_blocks, down_scales, down_bias) =
        down_bank.into_tensors(down, [experts, hidden, intermediate / 32]);
    Ok(ClampedRoutedExpertWeights::Native(Box::new(NativeExpertWeights {
        gate_up_blocks,
        gate_up_scales,
        gate_up_bias,
        down_blocks,
        down_scales,
        down_bias,
        marlin: Arc::new(Mutex::new(None)),
    })))
}

/// One OCP expert bank under construction: blocks, scales and bias.
struct Bank {
    blocks: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    bias: DeviceBuffer<bf16>,
    columns: usize,
}

impl Bank {
    fn new(backend: &CudaBackend, _name: &str, rows: usize, columns: usize) -> Result<Self> {
        let (pool, stream) = (&backend.inner.pool, &backend.inner.stream);
        Ok(Self {
            blocks: pool.allocate(stream, rows * columns / 2)?,
            scales: pool.allocate(stream, rows * columns / 32)?,
            bias: pool.allocate(stream, rows)?,
            columns,
        })
    }

    fn fill(
        &mut self,
        backend: &CudaBackend,
        repack: &GgmlMxFp4Repack,
        tensors: &CudaTensorSet,
        binding: &TensorBinding,
        placement: GgmlRowPlacement,
    ) -> Result<()> {
        let source = GgmlCheckpointWeight::load_binding(tensors, binding)?;
        if source.kind().ggml_type() != GgmlType::MXFP4 {
            return Err(Error::UnsupportedDecoderLayer(format!(
                "{}: GGML clamped-routed experts must be MXFP4",
                binding.source
            )));
        }
        let stream = &backend.inner.stream;
        let rows = source.rows();
        repack.blocks(
            stream,
            source.blocks(),
            &mut self.blocks,
            &mut self.scales,
            rows,
            self.columns,
            placement,
        )?;
        let bias = binding_bias(tensors, binding)?;
        let bias = bias.as_bf16().ok_or_else(|| Error::DTypeMismatch {
            name: bias.name().into(),
            expected: "BF16",
        })?;
        repack.bias(stream, bias, &mut self.bias, placement)
    }

    fn into_tensors(
        self,
        binding: &TensorBinding,
        shape: [usize; 3],
    ) -> (CudaTensor, CudaTensor, CudaTensor) {
        let [experts, rows, groups] = shape;
        let name = |part: &str| format!("{} ({part}, OCP)", binding.source);
        (
            CudaTensor::from_u8(name("blocks"), vec![experts, rows, groups, 16], self.blocks),
            CudaTensor::from_u8(name("scales"), shape.to_vec(), self.scales),
            CudaTensor::from_bf16(name("bias"), vec![experts, rows], self.bias),
        )
    }
}
